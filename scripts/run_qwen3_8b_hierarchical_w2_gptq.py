#!/usr/bin/env python3
"""Resumable true-sequential hierarchical-W2 GPTQ for Qwen3-8B."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
from fluxbin_style.hierarchical_w2 import (
    hierarchical_gptq_quantize,
    materialize_payload,
    nominal_bpw,
    payload_from_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"
GROUPS = (
    ("qkv", "self_attn.q_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("o", "self_attn.o_proj", ("self_attn.o_proj",)),
    ("gate_up", "mlp.gate_proj", ("mlp.gate_proj", "mlp.up_proj")),
    ("down", "mlp.down_proj", ("mlp.down_proj",)),
)
SOURCE_FILES = (
    "src/fluxbin_style/__init__.py",
    "src/fluxbin_style/evaluation.py",
    "src/fluxbin_style/hessian_obq.py",
    "src/fluxbin_style/hierarchical_w2.py",
    "src/fluxbin_style/qwen3.py",
    "src/fluxbin_style/qwen3_sequential.py",
    "scripts/run_qwen3_8b_hierarchical_w2_gptq.py",
)
PAYLOAD_FIELDS = {
    "q_packed", "parent", "r32_packed", "r16_packed",
    "r32_step", "r16_step", "permutation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--arm", choices=("H2.50", "H2.625", "H2.75", "H2.875"), required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_hash() -> tuple[str, dict[str, str]]:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode())
        digest.update(bytes.fromhex(value))
    return digest.hexdigest(), files


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema version")
    variants = config.get("variants", {})
    if list(variants) != ["H2.50", "H2.625", "H2.75", "H2.875"]:
        raise ValueError("the frozen four-arm inventory drifted")
    for name, value in variants.items():
        calculated = nominal_bpw(value["r32_bits"], value["r16_bits"])
        if calculated != value["nominal_bpw"]:
            raise ValueError(f"nominal bpw drifted for {name}")
    gptq = config["gptq"]
    expected = {
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
    if {key: gptq.get(key) for key in expected} != expected:
        raise ValueError("GPTQ controls drifted")
    if not all(config["policy"].values()):
        raise ValueError("an exclusion policy was disabled")


def validate_inputs(config: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
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


def validate_runtime(config: dict[str, Any]) -> torch.device:
    if list(sys.version_info[:2]) != config["execution"]["python_major_minor"]:
        raise RuntimeError(f"Python runtime drifted: {platform.python_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full calibration")
    torch.cuda.set_device(0)
    expected = config["execution"]
    device_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if device_name not in expected["quality_device_names"] or capability != expected["compute_capability"]:
        raise RuntimeError(f"formal A100 gate failed: {device_name}, capability={capability}")
    observed = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "safetensors": __import__("safetensors").__version__,
    }
    for name, value in observed.items():
        if value != expected[name]:
            raise RuntimeError(f"{name} runtime drifted: {value}")
    return torch.device("cuda:0")


def module_payload(tensors: dict[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    prefix = name + "."
    return {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}


def decode_module(
    tensors: dict[str, torch.Tensor],
    record: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    payload = payload_from_tensors(
        module_payload(tensors, record["module"]),
        out_features=record["shape"][0],
        in_features=record["shape"][1],
        r32_bits=record["r32_bits"],
        r16_bits=record["r16_bits"],
    )
    return materialize_payload(payload, device=device, dtype=torch.bfloat16)


def read_completed_layer(
    layer_dir: Path,
    *,
    arm: str,
    layer_index: int,
    config_hash: str,
    implementation_hash: str,
    layer: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    metadata = json.loads((layer_dir / "metadata.json").read_text())
    required = {
        "status": "passed",
        "arm": arm,
        "layer_index": layer_index,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"completed layer drifted: {layer_index}:{key}")
    path = layer_dir / "payload.safetensors"
    if sha256_file(path) != metadata["payload"]["sha256"]:
        raise ValueError("completed payload hash drifted")
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    if [record.get("module") for record in metadata.get("linears", [])] != list(QWEN3_LINEAR_MODULES):
        raise ValueError("completed Linear inventory drifted")
    expected_tensors = {
        f"{module}.{field}" for module in QWEN3_LINEAR_MODULES for field in PAYLOAD_FIELDS
    }
    if set(tensors) != expected_tensors:
        raise ValueError("completed payload tensor inventory drifted")
    for record in metadata["linears"]:
        layer.get_submodule(record["module"]).weight.copy_(
            decode_module(tensors, record, device=device)
        )
    return metadata


def write_layer(
    artifact_dir: Path,
    *,
    layer_index: int,
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final = artifact_dir / f"layer-{layer_index:03d}"
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f".layer-{layer_index:03d}.tmp-", dir=artifact_dir))
    path = temporary / "payload.safetensors"
    save_file({name: value.contiguous() for name, value in tensors.items()}, path)
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError("written tensor inventory drifted")
        for name, digest in tensor_hashes.items():
            if tensor_sha256(handle.get_tensor(name)) != digest:
                raise RuntimeError(f"written tensor drifted: {name}")
    completed = {
        **metadata,
        "payload": {
            "format": "hierarchical-w2-quality-v1",
            "path": "payload.safetensors",
            "file_bytes": path.stat().st_size,
            "semantic_tensor_bytes": sum(v.numel() * v.element_size() for v in tensors.values()),
            "sha256": sha256_file(path),
            "tensor_sha256": tensor_hashes,
            "round_trip_verified": True,
        },
    }
    atomic_json(temporary / "metadata.json", completed)
    os.replace(temporary, final)
    return completed


def quantize_module(
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    *,
    module_name: str,
    variant: dict[str, Any],
    config: dict[str, Any],
    payload_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target = module.weight.detach().to(torch.float32).clone()
    gptq = config["gptq"]
    projection = config["hierarchical_projection"]
    result = hierarchical_gptq_quantize(
        target,
        hessian,
        r32_bits=variant["r32_bits"],
        r16_bits=variant["r16_bits"],
        group_size=gptq["group_size"],
        block_size=gptq["block_size"],
        desc_act=gptq["desc_act"],
        damp_percent=gptq["damp_percent"],
        damp_auto_increment=gptq["damp_auto_increment"],
        alternating_rounds=projection["hierarchical_alternations"],
        step_multipliers=tuple(projection["relative_step_grid_multipliers"]),
    )
    module.weight.copy_(result.reconstructed.to(module.weight.dtype))
    for name, tensor in result.payload.tensors().items():
        payload_tensors[f"{module_name}.{name}"] = tensor
    target_norm = float(target.square().sum(dtype=torch.float64).item())
    return {
        "module": module_name,
        "shape": list(target.shape),
        "parameter_count": target.numel(),
        "r32_bits": variant["r32_bits"],
        "r16_bits": variant["r16_bits"],
        "semantic_tensor_bytes": result.payload.tensor_bytes,
        "actual_bpw": result.payload.actual_bpw,
        "squared_error": result.squared_error,
        "mean_squared_error": result.squared_error / target.numel(),
        "target_squared_norm": target_norm,
        "relative_frobenius_error": math.sqrt(result.squared_error / target_norm),
        "damping": result.damping,
        "damp_percent": result.damp_percent,
        "dead_columns": result.dead_columns,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    tokens = validate_inputs(config, args)
    if args.validate_only:
        print("HIERARCHICAL_W2_INPUTS_PASSED; no quantization launched", flush=True)
        return
    device = validate_runtime(config)
    implementation_hash, implementation_files = source_hash()
    config_hash = sha256_file(args.config)
    variant = config["variants"][args.arm]
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
            metadata = read_completed_layer(
                layer_dir,
                arm=args.arm,
                layer_index=layer_index,
                config_hash=config_hash,
                implementation_hash=implementation_hash,
                layer=layer,
                device=device,
            )
            resumed += 1
            print(f"HIERARCHICAL_W2_RESUMED={args.arm}:{layer_index}", flush=True)
        else:
            tensors: dict[str, torch.Tensor] = {}
            records: list[dict[str, Any]] = []
            captures: dict[str, Any] = {}
            for group_name, source_name, module_names in GROUPS:
                synchronize(device)
                capture_started = time.monotonic()
                capture = capture_linear_hessian(
                    layer, layer.get_submodule(source_name), inputs, forward_kwargs
                )
                synchronize(device)
                captures[group_name] = {
                    "source_module": source_name,
                    "shape": list(capture.hessian.shape),
                    "activation_rows": capture.activation_rows,
                    "diagonal_sha256": tensor_sha256(capture.hessian.diagonal()),
                    "elapsed_seconds": time.monotonic() - capture_started,
                }
                for module_name in module_names:
                    synchronize(device)
                    module_started = time.monotonic()
                    record = quantize_module(
                        layer.get_submodule(module_name),
                        capture.hessian,
                        module_name=module_name,
                        variant=variant,
                        config=config,
                        payload_tensors=tensors,
                    )
                    synchronize(device)
                    record["elapsed_seconds"] = time.monotonic() - module_started
                    records.append(record)
                    print(f"HIERARCHICAL_W2_QUANTIZED={args.arm}:{layer_index}:{module_name}", flush=True)
                del capture
                torch.cuda.empty_cache()
            metadata = write_layer(
                args.artifact_dir,
                layer_index=layer_index,
                metadata={
                    "schema_version": 1,
                    "status": "passed",
                    "arm": args.arm,
                    "layer_index": layer_index,
                    "config_sha256": config_hash,
                    "implementation_sha256": implementation_hash,
                    "model_revision": config["model"]["revision"],
                    "calibration_manifest_sha256": config["calibration"]["manifest_sha256"],
                    "true_sequential_group_order": [value[0] for value in GROUPS],
                    "hessians": captures,
                    "linears": records,
                    "elapsed_seconds_before_write": time.monotonic() - layer_started,
                },
                tensors=tensors,
            )
            newly_quantized += 1
            del tensors
            print(f"HIERARCHICAL_W2_COMMITTED={args.arm}:{layer_index}", flush=True)
        layers.append(metadata)
        propagate_layer_inputs(layer, inputs, outputs, forward_kwargs)
        inputs, outputs = outputs, inputs
        layer.to("cpu")
        torch.cuda.empty_cache()
        print(f"HIERARCHICAL_W2_LAYER={args.arm}:{layer_index + 1}/{layer_count}", flush=True)

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
        "arm": args.arm,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "implementation_files_sha256": implementation_files,
        "model": config["model"],
        "calibration": config["calibration"],
        "gptq": config["gptq"],
        "hierarchical_projection": config["hierarchical_projection"],
        "variant": variant,
        "coverage": {
            "layer_count": layer_count,
            "linear_count": len(records),
            "quantized_weight_count": parameters,
            "lm_head_quantized": False,
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
        },
        "quantization_time_seconds": projection_gptq_seconds,
        "time_breakdown_seconds": {
            "hessian_capture": hessian_capture_seconds,
            "hierarchical_projection_and_gptq": projection_gptq_seconds,
            "algorithm_total_excluding_model_load_and_io": hessian_capture_seconds + projection_gptq_seconds,
        },
        "execution": {
            "resumed_layers": resumed,
            "newly_quantized_layers": newly_quantized,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(0),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "artifacts": artifacts,
        "next_stage": "run_frozen_ppl" if complete else "resume_same_arm",
        "kernel_auto_launch": False,
        "rotation_auto_launch": False,
        "distillation_auto_launch": False,
    }
    atomic_json(args.output, result)
    print(f"HIERARCHICAL_W2_RESULT={args.output}", flush=True)


if __name__ == "__main__":
    main()
