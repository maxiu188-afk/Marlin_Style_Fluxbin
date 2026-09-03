#!/usr/bin/env python3
"""Resumable layer-sequential Qwen3 full-model Hessian OBQ quantization."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import (
    QWEN3_LINEAR_MODULES,
    TWO_BASE_PACKED_FORMAT,
    TwoBaseRankOneOptimizationConfig,
    atomic_json,
    capture_first_layer_inputs,
    capture_layer_hessians,
    expected_qwen3_linear_shape,
    invert_hessian,
    materialize_global_two_base_weight,
    materialize_hybrid_s8_weight,
    pack_two_bases,
    propagate_layer_inputs,
    quantize_hybrid_two_base_obq,
    quantize_pure_two_base_obq,
    sha256_file,
    tensor_sha256,
)


HESSIAN_GROUP_MODULES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}


IMPLEMENTATION_FILES = (
    "src/fluxbin_style/__init__.py",
    "src/fluxbin_style/evaluation.py",
    "src/fluxbin_style/hessian_obq.py",
    "src/fluxbin_style/packing.py",
    "src/fluxbin_style/qwen3.py",
    "src/fluxbin_style/qwen3_sequential.py",
    "src/fluxbin_style/residual_refinement.py",
    "src/fluxbin_style/two_base_rank1.py",
    "scripts/run_qwen3_full_hessian_obq_s8.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("pure", "hybrid_s8"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--single-linear-result", type=Path, required=True)
    parser.add_argument("--single-linear-payload", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-layer", type=int)
    return parser.parse_args()


def execution_layer_count(total_layers: int, stop_after_layer: int | None) -> int:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if stop_after_layer is None:
        return total_layers
    if not 0 <= stop_after_layer < total_layers:
        raise ValueError("stop_after_layer must identify an existing layer")
    return stop_after_layer + 1


def execution_layers(layers: Any, stop_after_layer: int | None) -> Any:
    """Return the single bounded layer view used by validation and execution."""
    return layers[: execution_layer_count(len(layers), stop_after_layer)]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_runtime(config: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    expected = config["execution"]
    if torch.cuda.get_device_name(0) != expected["device_name"]:
        raise RuntimeError("unexpected GPU")
    if list(torch.cuda.get_device_capability(0)) != expected["compute_capability"]:
        raise RuntimeError("unexpected compute capability")
    observed = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "safetensors": __import__("safetensors").__version__,
    }
    for name, version in observed.items():
        if version != expected[name]:
            raise RuntimeError(f"{name} runtime drifted: {version}")
    return torch.device("cuda:0")


def implementation_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_inputs(config: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    if config.get("schema_version") != 2:
        raise ValueError("unsupported schema_version")
    expected_groups = {
        name: list(modules) for name, modules in HESSIAN_GROUP_MODULES.items()
    }
    layerwise = config["layerwise_calibration"]
    if layerwise["exact_input_hessian_groups"] != expected_groups:
        raise ValueError("layerwise Hessian groups drifted")
    if layerwise["calibration_batch_size"] != 1:
        raise ValueError("only the accepted batch-one calibration path is supported")
    algorithm = config["algorithm"]
    if algorithm["global_num_bases"] != 2 or algorithm["global_group_size"] != 128:
        raise ValueError("global two-base contract drifted")
    if algorithm["hybrid_s8"]["residual_columns_per_group"] != 8:
        raise ValueError("hybrid-s8 contract drifted")
    if algorithm["arms_share_global_payload"]:
        raise ValueError("full-model arms must remain independent")
    if config["artifact_policy"]["payload_format"] != TWO_BASE_PACKED_FORMAT:
        raise ValueError("payload format drifted")
    model = config["model"]
    if args.snapshot_root.name != model["revision"]:
        raise ValueError("model snapshot revision drifted")
    accepted_calibration = config["accepted_calibration"]
    if sha256_file(args.calibration_manifest) != accepted_calibration["manifest_sha256"]:
        raise ValueError("accepted calibration manifest hash drifted")
    if sha256_file(args.calibration_tokens) != accepted_calibration["token_file_sha256"]:
        raise ValueError("accepted calibration token file hash drifted")
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("calibration manifest is not passed")
    if manifest["artifact_id"] != accepted_calibration["artifact_id"]:
        raise ValueError("calibration artifact id drifted")
    if manifest["config"]["dataset"]["revision"] != accepted_calibration[
        "dataset_revision"
    ]:
        raise ValueError("calibration dataset revision drifted")
    with safe_open(args.calibration_tokens, framework="pt", device="cpu") as artifact:
        if set(artifact.keys()) != {"token_ids"}:
            raise ValueError("calibration token inventory drifted")
        tokens = artifact.get_tensor("token_ids")
    if list(tokens.shape) != [
        accepted_calibration["sequences"],
        accepted_calibration["sequence_length"],
    ]:
        raise ValueError("calibration token shape drifted")
    if tensor_sha256(tokens) != accepted_calibration["token_tensor_sha256"]:
        raise ValueError("calibration token tensor hash drifted")

    gate = config["accepted_single_linear_gate"]
    if gate["outcome"] != "go_full_model":
        raise ValueError("single-Linear gate does not authorize full model")
    if sha256_file(args.single_linear_result) != gate["result_sha256"]:
        raise ValueError("single-Linear result hash drifted")
    if sha256_file(args.single_linear_payload) != gate["payload_sha256"]:
        raise ValueError("single-Linear payload hash drifted")
    single = json.loads(args.single_linear_result.read_text(encoding="utf-8"))
    if single.get("status") != "completed_pending_review":
        raise ValueError("single-Linear result status drifted")
    if single["reconstruction"]["pure_two_base_obq"][
        "calibration_total_output_squared_error"
    ] != gate["pure_calibration_loss"]:
        raise ValueError("single-Linear pure loss drifted")
    if single["reconstruction"]["hessian_salient_hybrid_s8_obq"][
        "calibration_total_output_squared_error"
    ] != gate["hybrid_calibration_loss"]:
        raise ValueError("single-Linear hybrid loss drifted")
    if not gate["hybrid_calibration_loss"] < gate["pure_calibration_loss"]:
        raise ValueError("single-Linear quality gate failed")
    return tokens.to(torch.int64)


def solver_config(value: dict[str, Any]) -> TwoBaseRankOneOptimizationConfig:
    return TwoBaseRankOneOptimizationConfig(**value)


def reconstruction_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    hessian: torch.Tensor,
) -> dict[str, float]:
    error = target.to(torch.float32) - reconstruction.to(torch.float32)
    squared_error = error.square().sum(dtype=torch.float64)
    target_squared_norm = target.square().sum(dtype=torch.float64)
    quadratic = 0.5 * (error @ hessian * error).sum(dtype=torch.float64)
    metrics = {
        "squared_error": float(squared_error.item()),
        "target_squared_norm": float(target_squared_norm.item()),
        "calibration_total_output_squared_error": float(quadratic.item()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("reconstruction metrics contain non-finite values")
    return metrics


def solver_summary(groups) -> dict[str, Any]:
    global_reasons = Counter(group.global_converged_reason for group in groups)
    refinement_reasons = Counter(
        group.refinement_converged_reason
        for group in groups
        if group.refinement_converged_reason is not None
    )
    return {
        "group_count": len(groups),
        "global_converged_reasons": dict(sorted(global_reasons.items())),
        "global_iterations": sum(group.global_iterations for group in groups),
        "refinement_converged_reasons": dict(sorted(refinement_reasons.items())),
        "refinement_iterations": sum(
            group.refinement_iterations or 0 for group in groups
        ),
        "propagated_update_squared_norm": math.fsum(
            group.propagated_update_squared_norm for group in groups
        ),
    }


@torch.no_grad()
def add_quantized_module(
    *,
    arm: str,
    module_name: str,
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    inverse_hessian: torch.Tensor,
    config: dict[str, Any],
    payload: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target_bf16 = module.weight.detach().clone()
    target = target_bf16.to(torch.float32)
    group_size = config["algorithm"]["global_group_size"]
    global_config = solver_config(config["global_solver"])
    if arm == "pure":
        result = quantize_pure_two_base_obq(
            target,
            inverse_hessian,
            group_size=group_size,
            config=global_config,
        )
        decomposition = result.decomposition
        payload[f"{module_name}.global_sign_codes"] = pack_two_bases(
            decomposition.bases
        ).cpu()
        payload[f"{module_name}.global_row_scales"] = decomposition.row_scales.cpu()
        payload[f"{module_name}.global_column_scales"] = (
            decomposition.column_scales.cpu()
        )
        sparse_maximum = None
    else:
        result = quantize_hybrid_two_base_obq(
            target,
            inverse_hessian,
            group_size=group_size,
            columns_per_group=config["algorithm"]["hybrid_s8"][
                "residual_columns_per_group"
            ],
            global_config=global_config,
            refinement_config=solver_config(config["refinement_solver"]),
        )
        decomposition = result.decomposition
        global_value = decomposition.global_decomposition
        refinement = decomposition.refinement_decomposition
        payload[f"{module_name}.global_sign_codes"] = pack_two_bases(
            global_value.bases
        ).cpu()
        payload[f"{module_name}.global_row_scales"] = global_value.row_scales.cpu()
        payload[f"{module_name}.global_column_scales"] = global_value.column_scales.cpu()
        payload[f"{module_name}.refinement_indices"] = (
            decomposition.selected_indices.to(torch.int16).cpu()
        )
        payload[f"{module_name}.refinement_sign_codes"] = pack_two_bases(
            refinement.bases
        ).cpu()
        payload[f"{module_name}.refinement_row_scales"] = refinement.row_scales.cpu()
        payload[f"{module_name}.refinement_column_scales"] = (
            refinement.column_scales.cpu()
        )
        delta = decomposition.refinement_grouped()
        mask = torch.zeros(
            (global_value.num_groups, group_size),
            dtype=torch.bool,
            device=target.device,
        )
        mask.scatter_(1, decomposition.selected_indices, True)
        outside = ~mask.unsqueeze(0).expand(target.shape[0], -1, -1)
        sparse_maximum = float(delta[outside].abs().max().item())
        if sparse_maximum != 0:
            raise RuntimeError("hybrid refinement changed an unselected column")
    # Use the packed-artifact materialization path immediately. This makes the
    # first pass, a resumed pass, and later PPL evaluation bit-identical after
    # the BF16 cast despite possible FP32 expression-order differences.
    reconstruction = materialize_module(
        payload,
        module_name,
        arm=arm,
        config=config,
        device=target.device,
    )
    metrics = reconstruction_metrics(target, reconstruction, hessian)
    module.weight.copy_(reconstruction)
    record = {
        "module": module_name,
        "shape": list(target.shape),
        "parameter_count": target.numel(),
        "target_bf16_sha256": tensor_sha256(target_bf16),
        "reconstruction": metrics,
        "solver": solver_summary(result.groups),
        "maximum_refinement_outside_selected_columns": sparse_maximum,
    }
    del target_bf16, target, reconstruction, result
    return record


def payload_subset(
    tensors: dict[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{module_name}."
    return {name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)}


def materialize_module(
    tensors: dict[str, torch.Tensor],
    module_name: str,
    *,
    arm: str,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    values = payload_subset(tensors, module_name)
    group_size = config["algorithm"]["global_group_size"]
    if arm == "pure":
        if set(values) != {
            "global_sign_codes",
            "global_row_scales",
            "global_column_scales",
        }:
            raise ValueError(f"pure payload inventory drifted for {module_name}")
        return materialize_global_two_base_weight(
            values["global_sign_codes"],
            values["global_row_scales"],
            values["global_column_scales"],
            group_size=group_size,
            device=device,
            output_dtype=torch.bfloat16,
        )
    if set(values) != {
        "global_sign_codes",
        "global_row_scales",
        "global_column_scales",
        "refinement_indices",
        "refinement_sign_codes",
        "refinement_row_scales",
        "refinement_column_scales",
    }:
        raise ValueError(f"hybrid payload inventory drifted for {module_name}")
    return materialize_hybrid_s8_weight(
        values["global_sign_codes"],
        values["global_row_scales"],
        values["global_column_scales"],
        values["refinement_indices"],
        values["refinement_sign_codes"],
        values["refinement_row_scales"],
        values["refinement_column_scales"],
        group_size=group_size,
        columns_per_group=config["algorithm"]["hybrid_s8"][
            "residual_columns_per_group"
        ],
        device=device,
        output_dtype=torch.bfloat16,
    )


@torch.no_grad()
def validate_and_apply_completed_layer(
    layer_dir: Path,
    *,
    arm: str,
    layer_index: int,
    layer: torch.nn.Module,
    config: dict[str, Any],
    config_hash: str,
    implementation_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    metadata_path = layer_dir / "metadata.json"
    payload_path = layer_dir / "payload.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "status": "passed",
        "arm": arm,
        "layer_index": layer_index,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "model_revision": config["model"]["revision"],
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"resumed layer metadata drifted: {layer_index}:{name}")
    if sha256_file(payload_path) != metadata["payload"]["sha256"]:
        raise ValueError(f"resumed layer payload hash drifted: {layer_index}")
    with safe_open(payload_path, framework="pt", device="cpu") as source:
        tensors = {name: source.get_tensor(name) for name in source.keys()}
    if set(tensors) != set(metadata["payload"]["tensor_sha256"]):
        raise ValueError(f"resumed layer payload inventory drifted: {layer_index}")
    for name, expected_hash in metadata["payload"]["tensor_sha256"].items():
        if tensor_sha256(tensors[name]) != expected_hash:
            raise ValueError(f"resumed tensor hash drifted: {layer_index}:{name}")
    for module_name in QWEN3_LINEAR_MODULES:
        module = layer.get_submodule(module_name)
        materialized = materialize_module(
            tensors,
            module_name,
            arm=arm,
            config=config,
            device=device,
        )
        if list(materialized.shape) != list(module.weight.shape):
            raise ValueError(f"resumed weight shape drifted: {layer_index}:{module_name}")
        module.weight.copy_(materialized)
    return metadata


def write_completed_layer(
    artifact_dir: Path,
    *,
    layer_index: int,
    metadata: dict[str, Any],
    payload: dict[str, torch.Tensor],
) -> tuple[Path, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_dir = artifact_dir / f"layer-{layer_index:03d}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed layer {final_dir}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".layer-{layer_index:03d}.tmp-", dir=artifact_dir)
    )
    payload_path = temporary / "payload.safetensors"
    save_file({name: value.contiguous() for name, value in payload.items()}, payload_path)
    tensor_hashes = {name: tensor_sha256(value) for name, value in payload.items()}
    with safe_open(payload_path, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(payload):
            raise RuntimeError("layer payload inventory drifted after write")
        for name, expected_hash in tensor_hashes.items():
            if tensor_sha256(reopened.get_tensor(name)) != expected_hash:
                raise RuntimeError(f"layer payload tensor hash drifted: {name}")
    completed = {
        **metadata,
        "payload": {
            "format": TWO_BASE_PACKED_FORMAT,
            "path": "payload.safetensors",
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
            "tensor_sha256": tensor_hashes,
            "round_trip_verified": True,
        },
    }
    atomic_json(temporary / "metadata.json", completed)
    os.replace(temporary, final_dir)
    return final_dir, completed


def aggregate_records(layer_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    linears = [record for layer in layer_metadata for record in layer["linears"]]
    squared_error = math.fsum(
        record["reconstruction"]["squared_error"] for record in linears
    )
    target_squared_norm = math.fsum(
        record["reconstruction"]["target_squared_norm"] for record in linears
    )
    calibration_loss = math.fsum(
        record["reconstruction"]["calibration_total_output_squared_error"]
        for record in linears
    )
    return {
        "layer_count": len(layer_metadata),
        "tensor_count": len(linears),
        "parameter_count": sum(record["parameter_count"] for record in linears),
        "squared_error": squared_error,
        "relative_frobenius_error": math.sqrt(squared_error / target_squared_norm),
        "calibration_total_output_squared_error": calibration_loss,
        "global_iterations": sum(
            record["solver"]["global_iterations"] for record in linears
        ),
        "refinement_iterations": sum(
            record["solver"]["refinement_iterations"] for record in linears
        ),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokens = validate_inputs(config, args)
    device = validate_runtime(config)
    config_hash = sha256_file(args.config)
    implementation_hash = implementation_sha256(args.project_root)
    torch.manual_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_config = json.loads((args.snapshot_root / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != config["model"]["model_type"]:
        raise ValueError("model type drifted")
    if len(model.model.layers) != config["model"]["expected_hidden_layers"]:
        raise ValueError("model layer count drifted")
    layers_to_execute = execution_layers(model.model.layers, args.stop_after_layer)
    layer_count = len(layers_to_execute)
    for layer_index, layer in enumerate(layers_to_execute):
        for module_name in QWEN3_LINEAR_MODULES:
            module = layer.get_submodule(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"target is not Linear: {layer_index}:{module_name}")
            if list(module.weight.shape) != expected_qwen3_linear_shape(
                model_config, module_name
            ):
                raise ValueError(f"target shape drifted: {layer_index}:{module_name}")

    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    model.model.layers[0].to(device)
    first = capture_first_layer_inputs(model, tokens, device=device)
    inputs = first.inputs
    forward_kwargs = first.forward_kwargs
    outputs = torch.empty_like(inputs)
    model.model.layers[0].to("cpu")
    model.model.embed_tokens.to("cpu")
    model.model.rotary_emb.to("cpu")
    del tokens
    torch.cuda.empty_cache()

    layer_metadata: list[dict[str, Any]] = []
    resumed_layers = 0
    newly_quantized_layers = 0
    for layer_index, layer in enumerate(layers_to_execute):
        layer_started = time.monotonic()
        layer.to(device)
        layer_dir = args.artifact_dir / f"layer-{layer_index:03d}"
        if layer_dir.is_dir():
            metadata = validate_and_apply_completed_layer(
                layer_dir,
                arm=args.arm,
                layer_index=layer_index,
                layer=layer,
                config=config,
                config_hash=config_hash,
                implementation_hash=implementation_hash,
                device=device,
            )
            resumed_layers += 1
            print(f"FLUXBIN_FULL_RESUMED_LAYER={layer_index}", flush=True)
        else:
            synchronize(device)
            capture_started = time.monotonic()
            capture = capture_layer_hessians(layer, inputs, forward_kwargs)
            synchronize(device)
            capture_elapsed = time.monotonic() - capture_started
            payload: dict[str, torch.Tensor] = {}
            linear_records: list[dict[str, Any]] = []
            hessian_records: dict[str, Any] = {}
            inversion_elapsed: dict[str, float] = {}
            linear_elapsed: dict[str, float] = {}
            for hessian_group, module_names in HESSIAN_GROUP_MODULES.items():
                hessian = capture.hessians.pop(hessian_group)
                synchronize(device)
                inversion_started = time.monotonic()
                inverse = invert_hessian(
                    hessian,
                    damp_percent=config["algorithm"]["damp_percent"],
                )
                synchronize(device)
                inversion_elapsed[hessian_group] = (
                    time.monotonic() - inversion_started
                )
                hessian_records[hessian_group] = {
                    "shape": list(hessian.shape),
                    "activation_rows": capture.activation_rows[hessian_group],
                    "diagonal_sha256": tensor_sha256(hessian.diagonal()),
                    "damping": inverse.damping,
                }
                for module_name in module_names:
                    synchronize(device)
                    linear_started = time.monotonic()
                    module = layer.get_submodule(module_name)
                    linear_records.append(
                        add_quantized_module(
                            arm=args.arm,
                            module_name=module_name,
                            module=module,
                            hessian=hessian,
                            inverse_hessian=inverse.inverse,
                            config=config,
                            payload=payload,
                        )
                    )
                    synchronize(device)
                    linear_elapsed[module_name] = time.monotonic() - linear_started
                    print(
                        f"FLUXBIN_FULL_QUANTIZED={args.arm}:"
                        f"{layer_index}:{module_name}",
                        flush=True,
                    )
                del hessian, inverse
                torch.cuda.empty_cache()
            metadata = {
                "schema_version": 2,
                "status": "passed",
                "arm": args.arm,
                "layer_index": layer_index,
                "config_sha256": config_hash,
                "implementation_sha256": implementation_hash,
                "model_revision": config["model"]["revision"],
                "calibration_manifest_sha256": config["accepted_calibration"][
                    "manifest_sha256"
                ],
                "hessians": hessian_records,
                "linears": linear_records,
                "stage_elapsed_seconds": {
                    "hessian_capture": capture_elapsed,
                    "hessian_inversion": inversion_elapsed,
                    "linear_quantization": linear_elapsed,
                },
                "elapsed_seconds_before_write": time.monotonic() - layer_started,
            }
            layer_dir, metadata = write_completed_layer(
                args.artifact_dir,
                layer_index=layer_index,
                metadata=metadata,
                payload=payload,
            )
            newly_quantized_layers += 1
            print(f"FLUXBIN_FULL_COMMITTED_LAYER={layer_index}", flush=True)
            del payload
        layer_metadata.append(metadata)
        propagate_layer_inputs(layer, inputs, outputs, forward_kwargs)
        inputs, outputs = outputs, inputs
        layer.to("cpu")
        torch.cuda.empty_cache()
        print(
            f"FLUXBIN_FULL_LAYER_COMPLETE={args.arm}:{layer_index + 1}/"
            f"{layer_count} elapsed={time.monotonic() - layer_started:.3f}",
            flush=True,
        )

    aggregate = aggregate_records(layer_metadata)
    expected = config["model"]
    expected_tensor_count = (
        expected["expected_tensor_count"] * layer_count
        // expected["expected_hidden_layers"]
    )
    expected_parameter_count = (
        expected["expected_parameter_count"] * layer_count
        // expected["expected_hidden_layers"]
    )
    if aggregate["tensor_count"] != expected_tensor_count:
        raise RuntimeError("full-model tensor count drifted")
    if aggregate["parameter_count"] != expected_parameter_count:
        raise RuntimeError("full-model parameter count drifted")
    layer_artifacts = []
    for layer_index, metadata in enumerate(layer_metadata):
        layer_dir = args.artifact_dir / f"layer-{layer_index:03d}"
        layer_artifacts.append(
            {
                "layer_index": layer_index,
                "metadata_path": str(layer_dir / "metadata.json"),
                "metadata_sha256": sha256_file(layer_dir / "metadata.json"),
                "payload_path": str(layer_dir / "payload.safetensors"),
                "payload_sha256": metadata["payload"]["sha256"],
                "payload_bytes": metadata["payload"]["bytes"],
            }
        )
    is_partial = layer_count < expected["expected_hidden_layers"]
    result = {
        "schema_version": 2,
        "status": (
            "partial_completed_pending_review"
            if is_partial
            else "completed_pending_review"
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": args.arm,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "model": config["model"],
        "calibration": config["accepted_calibration"],
        "algorithm": config["algorithm"],
        "layerwise_calibration": config["layerwise_calibration"],
        "aggregate": aggregate,
        "execution": {
            "resumed_layers": resumed_layers,
            "newly_quantized_layers": newly_quantized_layers,
            "requested_stop_after_layer": args.stop_after_layer,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "artifacts": layer_artifacts,
        "decision": {
            "execution_valid": True,
            "manual_review_required": True,
            "ppl_auto_launch": False,
            "backend_auto_launch": False,
        },
        "next_stage": "resume_same_artifact_dir" if is_partial else "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_FULL_RESULT={args.output}")


if __name__ == "__main__":
    main()
