#!/usr/bin/env python3
"""Evaluate sparse s=8 residual refinement on one accepted Qwen3 Linear."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import (
    NUM_BASES,
    TwoBaseRankOne,
    TwoBaseRankOneOptimizationConfig,
    atomic_json,
    error_metrics,
    gather_grouped_columns,
    optimize_sparse_residual_refinement,
    sha256_file,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--global-reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any], snapshot_root: Path) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    expected_algorithm = {
        "global_num_bases": 2,
        "global_group_size": 128,
        "base_values": [-1, 1],
        "row_column_scales_independent_per_base": True,
        "scale_dtype": "torch.float32",
        "joint_sign_pattern_count": 4,
        "residual_refinement": True,
        "residual_selection_metric": "column_squared_error",
        "residual_columns_per_group": 8,
        "residual_selection_ties": "stable_lower_index_first",
        "stored_indices_order": "ascending_within_group",
        "refinement_num_bases": 2,
        "refinement_group_size": 8,
        "hessian_error_propagation": False,
        "distillation": False,
    }
    if config["algorithm"] != expected_algorithm:
        raise ValueError("sparse residual-refinement contract drifted")
    if NUM_BASES != 2:
        raise ValueError("implementation base count drifted")
    model = config["model"]
    if model["model_type"] != "qwen3":
        raise ValueError("single-Linear gate requires Qwen3")
    if snapshot_root.name != model["revision"]:
        raise ValueError("snapshot path does not match pinned model revision")
    if config["global_reference"]["target_tensor_sha256"] != model["target_tensor_sha256"]:
        raise ValueError("global reference and target tensor differ")
    execution = config["execution"]
    if execution["device_name"] != "NVIDIA GH200 120GB":
        raise ValueError("execution device contract drifted")
    if config["decision"]["full_model_launch"] != "manual_review_only":
        raise ValueError("automatic full-model follow-up is forbidden")


def timed_cuda(callable_value):
    torch.cuda.synchronize()
    started = time.monotonic()
    value = callable_value()
    torch.cuda.synchronize()
    return value, time.monotonic() - started


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        temporary,
    )
    os.replace(temporary, path)


def load_and_validate_global_reference(
    result_path: Path,
    expected: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], TwoBaseRankOne]:
    if sha256_file(result_path) != expected["result_sha256"]:
        raise ValueError("accepted global result hash drifted")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "passed" or not result["decision"]["single_linear_execution_valid"]:
        raise ValueError("global reference is not an accepted result")
    if result["model"]["target_tensor_sha256"] != expected["target_tensor_sha256"]:
        raise ValueError("global reference target hash drifted")
    if result["source_manifest_sha256"] != expected["source_manifest_sha256"]:
        raise ValueError("global reference source manifest drifted")
    observed_sse = result["reconstruction"]["optimized_two_base_rank_one"]["squared_error"]
    if observed_sse != expected["optimized_squared_error"]:
        raise ValueError("global reference SSE drifted")
    payload_path = Path(result["payload"]["path"])
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    if sha256_file(payload_path) != expected["payload_sha256"]:
        raise ValueError("accepted global payload hash drifted")
    expected_names = {
        "two_base_signs",
        "two_base_row_scales",
        "two_base_column_scales",
    }
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(payload_path, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != expected_names:
            raise ValueError("global reference payload inventory drifted")
        for name in expected_names:
            value = reopened.get_tensor(name)
            if tensor_sha256(value) != result["payload"]["tensor_sha256"][name]:
                raise ValueError(f"global reference tensor hash drifted: {name}")
            tensors[name] = value.to(device=device)
    decomposition = TwoBaseRankOne(
        bases=tensors["two_base_signs"],
        row_scales=tensors["two_base_row_scales"],
        column_scales=tensors["two_base_column_scales"],
        group_size=128,
    )
    return result, decomposition


def main() -> None:
    args = parse_args()
    for output in (args.output, args.payload):
        if output.exists():
            raise FileExistsError(output)
    for required in (args.config, args.source_manifest, args.global_reference_result):
        if not required.is_file():
            raise FileNotFoundError(required)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config, args.snapshot_root)
    model = config["model"]
    execution = config["execution"]
    algorithm = config["algorithm"]
    target_name = model["target_tensor"]
    shard_path = args.snapshot_root / model["target_shard"]
    model_config_path = args.snapshot_root / "config.json"
    index_path = args.snapshot_root / "model.safetensors.index.json"
    for required in (shard_path, model_config_path, index_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    downloaded_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if downloaded_config.get("model_type") != model["model_type"]:
        raise ValueError("model type drifted")
    if index["weight_map"].get(target_name) != model["target_shard"]:
        raise ValueError("target shard mapping drifted")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != execution["device_name"]:
        raise RuntimeError("unexpected GPU")
    if list(torch.cuda.get_device_capability(0)) != execution["compute_capability"]:
        raise RuntimeError("unexpected compute capability")
    for package, observed in {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "safetensors": __import__("safetensors").__version__,
    }.items():
        if observed != execution[package]:
            raise RuntimeError(f"{package} runtime drifted: {observed}")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    device = torch.device("cuda:0")

    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        if target_name not in shard.keys():
            raise KeyError(target_name)
        checkpoint_weight = shard.get_tensor(target_name).contiguous()
    if list(checkpoint_weight.shape) != model["target_shape"]:
        raise ValueError("target shape drifted")
    if str(checkpoint_weight.dtype) != model["checkpoint_dtype"]:
        raise ValueError("target dtype drifted")
    if tensor_sha256(checkpoint_weight) != model["target_tensor_sha256"]:
        raise ValueError("target tensor hash drifted")
    target = checkpoint_weight.to(device=device, dtype=torch.float32)
    if not torch.isfinite(target).all():
        raise FloatingPointError("target contains non-finite values")

    global_result, global_decomposition = load_and_validate_global_reference(
        args.global_reference_result,
        config["global_reference"],
        device=device,
    )
    global_reconstruction = global_decomposition.reconstruct()
    global_metrics = error_metrics(target, global_reconstruction)
    expected_global_sse = float(config["global_reference"]["optimized_squared_error"])
    global_sse_tolerance = 1e-6 * max(expected_global_sse, 1.0)
    if abs(global_metrics["squared_error"] - expected_global_sse) > global_sse_tolerance:
        raise RuntimeError("reconstructed global SSE does not match accepted reference")

    refinement, refinement_seconds = timed_cuda(
        lambda: optimize_sparse_residual_refinement(
            target,
            global_decomposition,
            columns_per_group=int(algorithm["residual_columns_per_group"]),
            config=TwoBaseRankOneOptimizationConfig(**config["refinement_solver"]),
        )
    )
    hybrid = refinement.decomposition
    hybrid_reconstruction = hybrid.reconstruct()
    hybrid_metrics = error_metrics(target, hybrid_reconstruction)
    global_sse = global_metrics["squared_error"]
    hybrid_sse = hybrid_metrics["squared_error"]
    if not all(math.isfinite(value) for value in (*global_metrics.values(), *hybrid_metrics.values())):
        raise FloatingPointError("non-finite reconstruction metric")
    if not hybrid_sse < global_sse:
        raise RuntimeError("sparse residual refinement did not strictly improve global arm")

    group_size = int(algorithm["global_group_size"])
    columns_per_group = int(algorithm["residual_columns_per_group"])
    selected_indices = hybrid.selected_indices
    selected_global_residual = gather_grouped_columns(
        target - global_reconstruction,
        selected_indices,
        group_size=group_size,
    )
    selected_hybrid_residual = gather_grouped_columns(
        target - hybrid_reconstruction,
        selected_indices,
        group_size=group_size,
    )
    selected_before_sse = float(selected_global_residual.square().sum(dtype=torch.float64))
    selected_after_sse = float(selected_hybrid_residual.square().sum(dtype=torch.float64))
    global_grouped = global_reconstruction.reshape(
        global_decomposition.out_features,
        global_decomposition.num_groups,
        group_size,
    )
    hybrid_grouped = hybrid_reconstruction.reshape_as(global_grouped)
    selected_mask = torch.zeros(
        global_decomposition.num_groups,
        group_size,
        dtype=torch.bool,
        device=device,
    )
    selected_mask.scatter_(1, selected_indices, True)
    non_selected = (~selected_mask).unsqueeze(0).expand_as(global_grouped)
    non_selected_max_abs_delta = float(
        (hybrid_grouped - global_grouped)[non_selected].abs().max()
    )
    if non_selected_max_abs_delta != 0.0:
        raise RuntimeError("refinement changed a non-selected column")

    refinement_decomposition = hybrid.refinement_decomposition
    payload_tensors = {
        "global_signs": global_decomposition.bases,
        "global_row_scales": global_decomposition.row_scales,
        "global_column_scales": global_decomposition.column_scales,
        "refinement_indices": selected_indices.to(dtype=torch.int16),
        "refinement_signs": refinement_decomposition.bases,
        "refinement_row_scales": refinement_decomposition.row_scales,
        "refinement_column_scales": refinement_decomposition.column_scales,
    }
    payload_tensor_hashes = {
        name: tensor_sha256(value) for name, value in payload_tensors.items()
    }
    atomic_safetensors(args.payload, payload_tensors)
    with safe_open(args.payload, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(payload_tensors):
            raise RuntimeError("hybrid payload tensor inventory drifted")
        for name, expected_hash in payload_tensor_hashes.items():
            if tensor_sha256(reopened.get_tensor(name)) != expected_hash:
                raise RuntimeError(f"hybrid payload tensor hash drifted: {name}")

    out_features, in_features = model["target_shape"]
    num_groups = in_features // group_size
    weight_count = out_features * in_features
    global_binary_count = NUM_BASES * weight_count
    global_row_scale_count = NUM_BASES * out_features * num_groups
    global_column_scale_count = NUM_BASES * num_groups * group_size
    refinement_weight_count = out_features * num_groups * columns_per_group
    refinement_binary_count = NUM_BASES * refinement_weight_count
    refinement_row_scale_count = NUM_BASES * out_features * num_groups
    refinement_column_scale_count = NUM_BASES * num_groups * columns_per_group
    index_count = num_groups * columns_per_group
    global_scale_count = global_row_scale_count + global_column_scale_count
    refinement_scale_count = refinement_row_scale_count + refinement_column_scale_count
    global_fp32_bpw = (global_binary_count + global_scale_count * 32) / weight_count
    global_fp16_bpw = (global_binary_count + global_scale_count * 16) / weight_count
    hybrid_fp32_bpw = (
        global_binary_count
        + refinement_binary_count
        + (global_scale_count + refinement_scale_count) * 32
        + index_count * 16
    ) / weight_count
    hybrid_fp16_bpw = (
        global_binary_count
        + refinement_binary_count
        + (global_scale_count + refinement_scale_count) * 16
        + index_count * 16
    ) / weight_count
    relative_sse_reduction = (global_sse - hybrid_sse) / global_sse
    refinement_iterations = refinement.refinement_optimization.iterations
    result = {
        "schema_version": 1,
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "source_manifest_path": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "model": {
            **model,
            "snapshot_root": str(args.snapshot_root),
            "config_sha256": sha256_file(model_config_path),
            "index_sha256": sha256_file(index_path),
            "shard_path": str(shard_path),
            "shard_symlink_target": os.readlink(shard_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "safetensors": __import__("safetensors").__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        },
        "contract": algorithm,
        "global_reference": {
            **config["global_reference"],
            "result_path": str(args.global_reference_result),
            "payload_path": global_result["payload"]["path"],
            "payload_tensor_sha256": global_result["payload"]["tensor_sha256"],
            "validated": True,
        },
        "timing_seconds": {"sparse_residual_refinement": refinement_seconds},
        "reconstruction": {
            "global_two_base_rank_one": global_metrics,
            "hybrid_s8": hybrid_metrics,
            "squared_error_reduction_vs_global": global_sse - hybrid_sse,
            "relative_squared_error_reduction_vs_global": relative_sse_reduction,
            "strictly_improved_vs_global": True,
            "selected_residual_energy_fraction": refinement.selected_residual_energy_fraction,
            "selected_columns_squared_error_before": selected_before_sse,
            "selected_columns_squared_error_after": selected_after_sse,
            "selected_columns_relative_squared_error_reduction": (
                (selected_before_sse - selected_after_sse) / selected_before_sse
            ),
            "non_selected_max_abs_delta_vs_global": non_selected_max_abs_delta,
        },
        "selection": {
            "metric": algorithm["residual_selection_metric"],
            "columns_per_group": columns_per_group,
            "num_groups": num_groups,
            "index_dtype": "torch.int16",
            "indices_are_group_local": True,
            "indices_sorted_ascending": True,
            "indices_unique_within_group": True,
        },
        "storage": {
            "weight_count": weight_count,
            "global_binary_count": global_binary_count,
            "global_row_scale_count": global_row_scale_count,
            "global_column_scale_count": global_column_scale_count,
            "refinement_binary_count": refinement_binary_count,
            "refinement_row_scale_count": refinement_row_scale_count,
            "refinement_column_scale_count": refinement_column_scale_count,
            "refinement_index_count": index_count,
            "index_dtype_bits": 16,
            "algorithm_scale_dtype_bits": 32,
            "global_algorithm_effective_bits_per_weight": global_fp32_bpw,
            "global_projected_fp16_scale_effective_bits_per_weight": global_fp16_bpw,
            "hybrid_algorithm_effective_bits_per_weight": hybrid_fp32_bpw,
            "hybrid_projected_fp16_scale_effective_bits_per_weight": hybrid_fp16_bpw,
            "projection_is_not_a_runtime_measurement": True,
        },
        "refinement_solver": {
            "config": config["refinement_solver"],
            "initial_mse": refinement.refinement_optimization.initial_mse,
            "converged_reason": refinement.refinement_optimization.converged_reason,
            "iterations": [asdict(item) for item in refinement_iterations],
        },
        "payload": {
            "path": str(args.payload),
            "bytes": args.payload.stat().st_size,
            "sha256": sha256_file(args.payload),
            "tensor_sha256": payload_tensor_hashes,
        },
        "decision": {
            **config["decision"],
            "single_linear_execution_valid": True,
            "global_reference_validated": True,
            "strict_improvement_observed": True,
            "manual_review_required": True,
        },
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN2_S8_SINGLE_LINEAR_RESULT={args.output}")
    print(f"FLUXBIN2_S8_SINGLE_LINEAR_PAYLOAD={args.payload}")
    print(
        "FLUXBIN2_S8_SINGLE_LINEAR_EFFECT="
        f"global_sse={global_sse:.9f} hybrid_sse={hybrid_sse:.9f} "
        f"relative_reduction={relative_sse_reduction:.9f}"
    )
    print("FLUXBIN2_S8_SINGLE_LINEAR_PASSED")


if __name__ == "__main__":
    main()
