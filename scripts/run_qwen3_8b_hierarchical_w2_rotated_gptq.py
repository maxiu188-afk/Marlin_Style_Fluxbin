#!/usr/bin/env python3
"""Offline-rotated hierarchical-W2 GPTQ for Qwen3-8B, single H2.50 arm.

The rotation is folded into the BF16 checkpoint before any Hessian is
captured, so GPTQ sees the rotated basis and the calibration statistics belong
to the rotated model.  Everything downstream of that point is the accepted
unrotated pipeline, imported rather than copied, so the two runs differ only in
the rotation.

This script carries its own source-hash set and its own config, and it never
writes into the unrotated experiment's output root.  The accepted H2.50/H2.875
artifacts keep their `implementation_sha256`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from fluxbin_style import (
    QWEN3_LINEAR_MODULES,
    atomic_json,
    capture_first_layer_inputs,
    capture_linear_hessian,
    expected_qwen3_linear_shape,
    propagate_layer_inputs,
    sha256_file,
    tensor_sha256,
)
from fluxbin_style.hierarchical_w2 import nominal_bpw
from fluxbin_style.offline_rotation import (
    QWEN3_ROTATION_FORMAT,
    apply_offline_qwen3_rotation,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_8b_hierarchical_w2_gptq import (  # noqa: E402
    EXECUTION_POLICIES,
    FORMAL_EXECUTION_POLICY,
    GROUPS,
    quantize_module,
    read_completed_layer,
    validate_runtime,
    write_layer,
)
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
ARM = "H2.50"
SOURCE_FILES = (
    "src/fluxbin_style/__init__.py",
    "src/fluxbin_style/evaluation.py",
    "src/fluxbin_style/hessian_obq.py",
    "src/fluxbin_style/hierarchical_w2.py",
    "src/fluxbin_style/offline_rotation.py",
    "src/fluxbin_style/qwen3.py",
    "src/fluxbin_style/qwen3_sequential.py",
    "scripts/run_qwen3_8b_hierarchical_w2_gptq.py",
    "scripts/run_qwen3_8b_hierarchical_w2_rotated_gptq.py",
)
FROZEN_GPTQ_CONTROLS = {
    "group_size": 128,
    "block_size": 128,
    "desc_act": True,
    "act_group_aware": False,
    "static_groups": False,
    "true_sequential": True,
    "lm_head": False,
    "mse": 0.0,
    "damp_percent": 0.01,
    "damp_auto_increment": 0.01,
    "fallback": None,
}
DECLARED_ROTATION_FIELDS = (
    "rotate_values",
    "online_mlp_hadamard",
    "online_qk_hadamard",
    "down_proj_input_rotated",
    "activation_quantization",
    "custom_kernel",
)
RUNTIME_IDENTITY_FIELDS = (
    "execution_policy",
    "formal_a100_device_match",
    "device",
    "compute_capability",
    "python",
    "torch",
    "transformers",
    "datasets",
    "safetensors",
    "cuda",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-policy", choices=EXECUTION_POLICIES, default=FORMAL_EXECUTION_POLICY
    )
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def source_hash() -> tuple[str, dict[str, str]]:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode())
        digest.update(bytes.fromhex(value))
    return digest.hexdigest(), files


def runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must remain identical across a resumed arm."""

    missing = [key for key in RUNTIME_IDENTITY_FIELDS if key not in runtime]
    if missing:
        raise ValueError(f"runtime identity is incomplete: {missing}")
    return {key: runtime[key] for key in RUNTIME_IDENTITY_FIELDS}


def bind_runtime_identity(artifact_dir: Path, identity: dict[str, Any]) -> None:
    """Bind an artifact root before any layer can be written or resumed."""

    path = artifact_dir / "runtime-identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError("artifact root belongs to a different runtime identity")
        return
    if artifact_dir.is_dir() and any(artifact_dir.glob("layer-*")):
        raise RuntimeError("existing layers have no runtime identity; refusing mixed resume")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(path, identity)


def validate_layer_runtime_identity(
    layer_dir: Path, expected: dict[str, Any]
) -> None:
    metadata = json.loads((layer_dir / "metadata.json").read_text())
    if metadata.get("runtime_identity") != expected:
        raise ValueError(f"completed layer runtime drifted: {layer_dir.name}")


def validate_config(config: dict[str, Any]) -> None:
    """Gate the rotated experiment's own frozen shape."""

    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema version")
    if config.get("experiment_id") != "qwen3-8b-hierarchical-w2-rotated-v1":
        raise ValueError("rotated experiment id drifted")
    variants = config.get("variants", {})
    if list(variants) != [ARM]:
        raise ValueError("the rotated study is a single-arm experiment")
    variant = variants[ARM]
    if nominal_bpw(variant["r32_bits"], variant["r16_bits"]) != variant["nominal_bpw"]:
        raise ValueError("nominal bpw drifted")
    gptq = config["gptq"]
    if {key: gptq.get(key) for key in FROZEN_GPTQ_CONTROLS} != FROZEN_GPTQ_CONTROLS:
        raise ValueError("GPTQ controls drifted from the unrotated baseline")
    rotation = config["rotation"]
    if rotation.get("format") != QWEN3_ROTATION_FORMAT:
        raise ValueError("rotation format drifted")
    for key, expected in (
        ("rotate_values", True),
        ("online_mlp_hadamard", False),
        ("online_qk_hadamard", False),
        ("down_proj_input_rotated", False),
        ("activation_quantization", False),
        ("custom_kernel", False),
        ("learned_rotation", False),
    ):
        if rotation.get(key) is not expected:
            raise ValueError(f"rotation control drifted: {key}")
    if config["evaluation"]["arms"] != ["gptq_w3_g128_sym", ARM]:
        raise ValueError("evaluation arms drifted")
    if not all(config["policy"].values()):
        raise ValueError("an exclusion policy was disabled")
    if not config["policy"].get("offline_rotation_only"):
        raise ValueError("this runner only implements the offline rotation route")
    if config["execution"].get("cross_device_quality") != {
        "device_name_contains": "RTX PRO 4500",
        "compute_capability": [12, 0],
        "status": "cross_device_report_only",
    }:
        raise ValueError("cross-device quality route drifted")
    baselines = config["baselines"]
    for key in (
        "bf16_reference_perplexity",
        "w3_endpoint_same_run_perplexity",
        "unrotated_h2_50_perplexity",
    ):
        if not isinstance(baselines.get(key), float):
            raise ValueError(f"comparison baseline missing: {key}")
    if sha256_file(ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_v1.json") != baselines[
        "unrotated_source_config_sha256"
    ]:
        raise ValueError("the recorded unrotated baseline config drifted")


def validate_inputs(config: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    """Same artifact gate as the unrotated runner, over the rotated config."""

    from safetensors import safe_open

    validate_config(config)
    model = config["model"]
    if args.snapshot_root.name != model["revision"]:
        raise ValueError("model revision drifted")
    model_config = json.loads((args.snapshot_root / "config.json").read_text())
    if model_config.get("model_type") != "qwen3" or model_config.get("num_hidden_layers") != 36:
        raise ValueError("model architecture drifted")
    for name, digest in model["model_preflight_files"].items():
        if sha256_file(args.snapshot_root / name) != digest:
            raise ValueError(f"snapshot content drifted: {name}")
    calibration = config["calibration"]
    if sha256_file(args.calibration_manifest) != calibration["manifest_sha256"]:
        raise ValueError("calibration manifest drifted")
    if sha256_file(args.calibration_tokens) != calibration["token_file_sha256"]:
        raise ValueError("calibration token file drifted")
    with safe_open(args.calibration_tokens, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"token_ids"}:
            raise ValueError("calibration tensor inventory drifted")
        tokens = handle.get_tensor("token_ids")
    if list(tokens.shape) != [calibration["sample_count"], calibration["sequence_length"]]:
        raise ValueError("calibration token shape drifted")
    if tensor_sha256(tokens) != calibration["token_tensor_sha256"]:
        raise ValueError("calibration token values drifted")
    return tokens.to(torch.int64)


def rotate_model(
    model: torch.nn.Module, config: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    """Fold the declared rotation in and check it against the frozen config."""

    declared = config["rotation"]
    record = apply_offline_qwen3_rotation(
        model, rotate_values=declared["rotate_values"], compute_device=device
    )
    if record["format"] != declared["format"]:
        raise RuntimeError("applied rotation format differs from the config")
    for key in DECLARED_ROTATION_FIELDS:
        if record[key] != declared[key]:
            raise RuntimeError(f"applied rotation differs from the config: {key}")
    for key, expected in (
        ("residual_size", declared["residual_size"]),
        ("head_rotation_size", declared["head_rotation_size"]),
    ):
        if record[key] != expected:
            raise RuntimeError(f"applied rotation geometry differs: {key}")
    return record


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    tokens = validate_inputs(config, args)
    if args.validate_only:
        print("ROTATED_W2_INPUTS_PASSED; no quantization launched", flush=True)
        return
    device, runtime = validate_runtime(config, execution_policy=args.execution_policy)
    runtime = {**runtime, "cuda": torch.version.cuda}
    identity = runtime_identity(runtime)
    bind_runtime_identity(args.artifact_dir, identity)
    implementation_hash, implementation_files = source_hash()
    config_hash = sha256_file(args.config)
    variant = config["variants"][ARM]
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=config["evaluation"]["attention_implementation"],
    ).eval()
    model.config.use_cache = False
    model_config = json.loads((args.snapshot_root / "config.json").read_text())
    stop = args.stop_after_layer
    layer_count = len(model.model.layers) if stop is None else stop + 1
    if layer_count <= 0 or layer_count > len(model.model.layers):
        raise ValueError("invalid stop-after-layer")
    for layer in model.model.layers[:layer_count]:
        for name in QWEN3_LINEAR_MODULES:
            module = layer.get_submodule(name)
            if not isinstance(module, torch.nn.Linear) or list(module.weight.shape) != expected_qwen3_linear_shape(model_config, name):
                raise ValueError(f"Linear inventory drifted: {name}")

    rotation_started = time.monotonic()
    rotation = rotate_model(model, config, device=device)
    rotation["elapsed_seconds"] = time.monotonic() - rotation_started
    torch.cuda.empty_cache()
    print(f"ROTATED_W2_ROTATION_APPLIED={rotation['format']}", flush=True)

    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    model.model.layers[0].to(device)
    first = capture_first_layer_inputs(model, tokens, device=device)
    inputs, forward_kwargs = first.inputs, first.forward_kwargs
    outputs = torch.empty_like(inputs)
    model.model.layers[0].to("cpu")
    model.model.embed_tokens.to("cpu")
    model.model.rotary_emb.to("cpu")
    del tokens
    torch.cuda.empty_cache()

    layers: list[dict[str, Any]] = []
    resumed = 0
    newly_quantized = 0
    for layer_index, layer in enumerate(model.model.layers[:layer_count]):
        layer_started = time.monotonic()
        layer.to(device)
        layer_dir = args.artifact_dir / f"layer-{layer_index:03d}"
        if layer_dir.is_dir():
            validate_layer_runtime_identity(layer_dir, identity)
            metadata = read_completed_layer(
                layer_dir,
                arm=ARM,
                layer_index=layer_index,
                config_hash=config_hash,
                implementation_hash=implementation_hash,
                layer=layer,
                device=device,
            )
            resumed += 1
            print(f"ROTATED_W2_RESUMED={layer_index}", flush=True)
        else:
            tensors: dict[str, torch.Tensor] = {}
            records: list[dict[str, Any]] = []
            captures: dict[str, Any] = {}
            for group_name, source_name, module_names in GROUPS:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                capture_started = time.monotonic()
                capture = capture_linear_hessian(
                    layer, layer.get_submodule(source_name), inputs, forward_kwargs
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                captures[group_name] = {
                    "source_module": source_name,
                    "shape": list(capture.hessian.shape),
                    "activation_rows": capture.activation_rows,
                    "diagonal_sha256": tensor_sha256(capture.hessian.diagonal()),
                    "elapsed_seconds": time.monotonic() - capture_started,
                }
                for module_name in module_names:
                    module_started = time.monotonic()
                    record = quantize_module(
                        layer.get_submodule(module_name),
                        capture.hessian,
                        module_name=module_name,
                        variant=variant,
                        config=config,
                        payload_tensors=tensors,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    record["elapsed_seconds"] = time.monotonic() - module_started
                    records.append(record)
                    print(f"ROTATED_W2_QUANTIZED={layer_index}:{module_name}", flush=True)
                del capture
                torch.cuda.empty_cache()
            metadata = write_layer(
                args.artifact_dir,
                layer_index=layer_index,
                metadata={
                    "schema_version": 1,
                    "status": "passed",
                    "arm": ARM,
                    "layer_index": layer_index,
                    "config_sha256": config_hash,
                    "implementation_sha256": implementation_hash,
                    "model_revision": config["model"]["revision"],
                    "calibration_manifest_sha256": config["calibration"]["manifest_sha256"],
                    "rotation_format": rotation["format"],
                    "runtime_identity": identity,
                    "true_sequential_group_order": [value[0] for value in GROUPS],
                    "hessians": captures,
                    "linears": records,
                    "elapsed_seconds_before_write": time.monotonic() - layer_started,
                },
                tensors=tensors,
            )
            newly_quantized += 1
            del tensors
            print(f"ROTATED_W2_COMMITTED={layer_index}", flush=True)
        layers.append(metadata)
        propagate_layer_inputs(layer, inputs, outputs, forward_kwargs)
        inputs, outputs = outputs, inputs
        layer.to("cpu")
        torch.cuda.empty_cache()
        print(f"ROTATED_W2_LAYER={layer_index + 1}/{layer_count}", flush=True)

    records = [record for layer in layers for record in layer["linears"]]
    parameters = sum(record["parameter_count"] for record in records)
    tensor_bytes = sum(record["semantic_tensor_bytes"] for record in records)
    squared_error = math.fsum(record["squared_error"] for record in records)
    target_norm = math.fsum(record["target_squared_norm"] for record in records)
    hessian_capture_seconds = math.fsum(
        capture["elapsed_seconds"]
        for layer in layers
        for capture in layer["hessians"].values()
    )
    projection_gptq_seconds = math.fsum(record["elapsed_seconds"] for record in records)
    complete = layer_count == config["model"]["expected_hidden_layers"]
    expected_linears = config["model"]["expected_tensor_count"] * layer_count // 36
    expected_parameters = config["model"]["expected_parameter_count"] * layer_count // 36
    if len(records) != expected_linears or parameters != expected_parameters:
        raise RuntimeError("coverage drifted")
    artifacts = []
    for index, metadata in enumerate(layers):
        directory = args.artifact_dir / f"layer-{index:03d}"
        artifacts.append({
            "layer_index": index,
            "metadata": str(directory / "metadata.json"),
            "metadata_sha256": sha256_file(directory / "metadata.json"),
            "payload": str(directory / "payload.safetensors"),
            "payload_sha256": metadata["payload"]["sha256"],
        })
    result = {
        "schema_version": 1,
        "status": "completed_pending_ppl" if complete else "partial_completed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": ARM,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "implementation_files_sha256": implementation_files,
        "model": config["model"],
        "calibration": config["calibration"],
        "gptq": config["gptq"],
        "hierarchical_projection": config["hierarchical_projection"],
        "rotation": rotation,
        "variant": variant,
        "baselines": config["baselines"],
        "coverage": {
            "layer_count": layer_count,
            "linear_count": len(records),
            "quantized_weight_count": parameters,
            "lm_head_quantized": False,
            "lm_head_reparameterized": True,
            "no_fallback": True,
        },
        "storage": {
            "nominal_bpw": variant["nominal_bpw"],
            "semantic_tensor_bytes_including_permutation_and_steps": tensor_bytes,
            "actual_bpw_including_permutation_and_steps": 8.0 * tensor_bytes / parameters,
        },
        "reconstruction": {
            "squared_error": squared_error,
            "mean_squared_error": squared_error / parameters,
            "relative_frobenius_error": math.sqrt(squared_error / target_norm),
            "measured_in_rotated_basis": True,
        },
        "quantization_time_seconds": projection_gptq_seconds,
        "time_breakdown_seconds": {
            "offline_rotation": rotation["elapsed_seconds"],
            "hessian_capture": hessian_capture_seconds,
            "hierarchical_projection_and_gptq": projection_gptq_seconds,
            "algorithm_total_excluding_model_load_and_io": (
                rotation["elapsed_seconds"] + hessian_capture_seconds + projection_gptq_seconds
            ),
        },
        "execution": {
            **runtime,
            "resumed_layers": resumed,
            "newly_quantized_layers": newly_quantized,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "host": platform.node(),
        },
        "artifacts": artifacts,
        "next_stage": "run_rotated_ppl" if complete else "resume_rotated_arm",
        "kernel_auto_launch": False,
        "online_hadamard_auto_launch": False,
        "distillation_auto_launch": False,
    }
    atomic_json(args.output, result)
    print(f"ROTATED_W2_RESULT={args.output}", flush=True)


if __name__ == "__main__":
    main()
