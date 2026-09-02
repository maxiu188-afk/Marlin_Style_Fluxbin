#!/usr/bin/env python3
"""Run the frozen two-base rank-one gate on one real Qwen3 Linear."""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    TwoBaseRankOneOptimizationConfig,
    atomic_json,
    error_metrics,
    initialize_two_base_rank_one,
    optimize_two_base_rank_one,
    sha256_file,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any], snapshot_root: Path) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    algorithm = config["algorithm"]
    expected_algorithm = {
        "num_bases": 2,
        "group_size": 128,
        "base_values": [-1, 1],
        "row_column_scales_independent_per_base": True,
        "scale_dtype": "torch.float32",
        "joint_sign_pattern_count": 4,
        "salient_refinement": False,
        "compensation_matrix": False,
        "hessian_error_propagation": False,
        "distillation": False,
    }
    if algorithm != expected_algorithm:
        raise ValueError("two-base algorithm contract drifted")
    if NUM_BASES != algorithm["num_bases"]:
        raise ValueError("implementation base count drifted")
    model = config["model"]
    if model["model_type"] != "qwen3":
        raise ValueError("single-Linear gate requires Qwen3")
    if snapshot_root.name != model["revision"]:
        raise ValueError("snapshot path does not match pinned model revision")
    execution = config["execution"]
    if execution["device_name"] != "NVIDIA GH200 120GB":
        raise ValueError("execution device contract drifted")
    if config["decision"]["full_model_launch"] != "manual_review_only":
        raise ValueError("automatic follow-up is forbidden")


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


def main() -> None:
    args = parse_args()
    for output in (args.output, args.payload):
        if output.exists():
            raise FileExistsError(output)
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config, args.snapshot_root)
    model = config["model"]
    execution = config["execution"]
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

    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        if target_name not in shard.keys():
            raise KeyError(target_name)
        checkpoint_weight = shard.get_tensor(target_name).contiguous()
    if list(checkpoint_weight.shape) != model["target_shape"]:
        raise ValueError("target shape drifted")
    if str(checkpoint_weight.dtype) != model["checkpoint_dtype"]:
        raise ValueError("target dtype drifted")
    checkpoint_hash = tensor_sha256(checkpoint_weight)
    if checkpoint_hash != model["target_tensor_sha256"]:
        raise ValueError("target tensor hash drifted")
    target = checkpoint_weight.to(device="cuda:0", dtype=torch.float32)
    if not torch.isfinite(target).all():
        raise FloatingPointError("target contains non-finite values")

    group_size = int(config["algorithm"]["group_size"])
    initial, initial_seconds = timed_cuda(
        lambda: initialize_two_base_rank_one(target, group_size=group_size)
    )
    optimization, optimization_seconds = timed_cuda(
        lambda: optimize_two_base_rank_one(
            target,
            initial,
            TwoBaseRankOneOptimizationConfig(**config["solver"]),
        )
    )
    initial_metrics = error_metrics(target, initial.reconstruct())
    final_metrics = error_metrics(target, optimization.decomposition.reconstruct())
    initial_sse = initial_metrics["squared_error"]
    final_sse = final_metrics["squared_error"]
    allowed = float(config["solver"]["monotonicity_tolerance"]) * max(
        initial_sse,
        torch.finfo(torch.float32).eps,
    )
    if final_sse > initial_sse + allowed:
        raise RuntimeError("optimized decomposition regressed from greedy parent")

    decomposition = optimization.decomposition
    payload_tensors = {
        "two_base_signs": decomposition.bases,
        "two_base_row_scales": decomposition.row_scales,
        "two_base_column_scales": decomposition.column_scales,
    }
    payload_tensor_hashes = {
        name: tensor_sha256(value) for name, value in payload_tensors.items()
    }
    atomic_safetensors(args.payload, payload_tensors)
    with safe_open(args.payload, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(payload_tensors):
            raise RuntimeError("payload tensor inventory drifted")
        for name, expected_hash in payload_tensor_hashes.items():
            if tensor_sha256(reopened.get_tensor(name)) != expected_hash:
                raise RuntimeError(f"payload tensor hash drifted: {name}")

    out_features, in_features = model["target_shape"]
    num_groups = in_features // group_size
    weight_count = out_features * in_features
    row_scale_count = NUM_BASES * out_features * num_groups
    column_scale_count = NUM_BASES * num_groups * group_size
    scale_count = row_scale_count + column_scale_count
    algorithm_bits = NUM_BASES + scale_count * 32 / weight_count
    projected_fp16_bits = NUM_BASES + scale_count * 16 / weight_count
    relative_reduction = (initial_sse - final_sse) / initial_sse
    strict_improvement = final_sse < initial_sse
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
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        },
        "contract": config["algorithm"],
        "timing_seconds": {
            "greedy_initialization": initial_seconds,
            "rank_one_optimization": optimization_seconds,
        },
        "reconstruction": {
            "greedy_two_base": initial_metrics,
            "optimized_two_base_rank_one": final_metrics,
            "squared_error_reduction": initial_sse - final_sse,
            "relative_squared_error_reduction": relative_reduction,
            "strictly_improved": strict_improvement,
        },
        "storage": {
            "weight_count": weight_count,
            "logical_binary_bits_per_weight": NUM_BASES,
            "row_scale_count": row_scale_count,
            "column_scale_count": column_scale_count,
            "algorithm_scale_dtype_bits": 32,
            "algorithm_effective_bits_per_weight": algorithm_bits,
            "projected_fp16_scale_effective_bits_per_weight": projected_fp16_bits,
            "projection_is_not_a_runtime_measurement": True,
        },
        "solver": {
            "config": config["solver"],
            "initial_mse": optimization.initial_mse,
            "converged_reason": optimization.converged_reason,
            "iterations": [asdict(item) for item in optimization.iterations],
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
            "strict_improvement_observed": strict_improvement,
            "manual_review_required": True,
        },
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN2_SINGLE_LINEAR_RESULT={args.output}")
    print(f"FLUXBIN2_SINGLE_LINEAR_PAYLOAD={args.payload}")
    print(
        "FLUXBIN2_SINGLE_LINEAR_EFFECT="
        f"initial_sse={initial_sse:.9f} final_sse={final_sse:.9f} "
        f"relative_reduction={relative_reduction:.9f}"
    )
    print("FLUXBIN2_SINGLE_LINEAR_PASSED")


if __name__ == "__main__":
    main()
