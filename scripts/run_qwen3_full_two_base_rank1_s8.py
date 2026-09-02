#!/usr/bin/env python3
"""Resumable full-model two-base plus sparse-s8 reconstruction for Qwen3."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import (
    NUM_BASES,
    TWO_BASE_PACKED_FORMAT,
    TwoBaseRankOneOptimizationConfig,
    atomic_json,
    build_qwen3_linear_inventory,
    gather_grouped_columns,
    initialize_two_base_rank_one,
    optimize_sparse_residual_refinement,
    optimize_two_base_rank_one,
    pack_two_bases,
    parse_qwen3_linear_name,
    qwen3_linear_weight_names,
    sha256_file,
    tensor_sha256,
    unpack_two_bases,
)


IMPLEMENTATION_FILES = (
    "src/fluxbin_style/__init__.py",
    "src/fluxbin_style/evaluation.py",
    "src/fluxbin_style/packing.py",
    "src/fluxbin_style/qwen3.py",
    "src/fluxbin_style/residual_refinement.py",
    "src/fluxbin_style/two_base_rank1.py",
    "scripts/run_qwen3_full_two_base_rank1_s8.py",
)


MODEL_CONTRACT = {
    "repo_id": "Qwen/Qwen3-32B",
    "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "model_type": "qwen3",
    "checkpoint_dtype": "torch.bfloat16",
    "expected_hidden_layers": 64,
    "expected_tensor_count": 448,
    "expected_parameter_count": 31_205_621_760,
    "expected_weight_shards": 17,
    "expected_checkpoint_bytes": 65_524_246_528,
}


ALGORITHM_CONTRACT = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--single-linear-result", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any], snapshot_root: Path) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if config["model"] != MODEL_CONTRACT:
        raise ValueError("Qwen3-32B full-model contract drifted")
    if snapshot_root.name != MODEL_CONTRACT["revision"]:
        raise ValueError("snapshot path does not match pinned revision")
    if config["algorithm"] != ALGORITHM_CONTRACT:
        raise ValueError("two-base hybrid-s8 algorithm contract drifted")
    gate = config["single_linear_gate"]
    if gate["outcome"] != "go_full_model" or not (
        gate["hybrid_squared_error"] < gate["global_squared_error"]
    ):
        raise ValueError("single-Linear gate does not authorize full-model work")
    policy = config["artifact_policy"]
    if policy != {
        "artifact_id": "qwen3-32b-full-two-base-rank1-s8-v1",
        "payload_format": TWO_BASE_PACKED_FORMAT,
        "one_payload_per_tensor": True,
        "resumable": True,
        "share_global_arm_between_pure_and_hybrid": True,
        "stored_arms": [
            "global_two_base_rank_one",
            "sparse_residual_refinement_s8",
        ],
    }:
        raise ValueError("artifact policy drifted")
    decision = config["decision"]
    if not decision["require_every_tensor_strict_hybrid_improvement"]:
        raise ValueError("per-tensor hybrid improvement gate is required")
    if not decision["require_exact_zero_outside_selected_columns"]:
        raise ValueError("non-selected zero-delta gate is required")
    if decision["ppl_auto_launch"] or decision["backend_auto_launch"]:
        raise ValueError("downstream auto-launch is forbidden")


def validate_single_linear_gate(path: Path, expected: dict[str, Any]) -> None:
    if sha256_file(path) != expected["result_sha256"]:
        raise ValueError("single-Linear result hash drifted")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "passed" or not result["decision"]["single_linear_execution_valid"]:
        raise ValueError("single-Linear result is not accepted")
    if result["model"]["target_tensor"] != expected["target_tensor"]:
        raise ValueError("single-Linear target name drifted")
    if result["model"]["target_tensor_sha256"] != expected["target_tensor_sha256"]:
        raise ValueError("single-Linear target hash drifted")
    if result["payload"]["sha256"] != expected["payload_sha256"]:
        raise ValueError("single-Linear payload hash drifted")
    reconstruction = result["reconstruction"]
    if reconstruction["global_two_base_rank_one"]["squared_error"] != expected[
        "global_squared_error"
    ]:
        raise ValueError("single-Linear global SSE drifted")
    if reconstruction["hybrid_s8"]["squared_error"] != expected["hybrid_squared_error"]:
        raise ValueError("single-Linear hybrid SSE drifted")


def implementation_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def snapshot_inventory(
    snapshot_root: Path,
    model_config: dict[str, Any],
    index: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight_map")
    shard_names = tuple(sorted(set(weight_map.values())))
    if len(shard_names) != MODEL_CONTRACT["expected_weight_shards"]:
        raise ValueError("weight shard count drifted")
    records: list[dict[str, Any]] = []
    for shard_name in shard_names:
        shard_path = snapshot_root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in sorted(shard.keys()):
                if weight_map.get(name) != shard_name:
                    raise ValueError(f"index mapping drifted for {name}")
                records.append(
                    {
                        "name": name,
                        "shape": list(shard.get_slice(name).get_shape()),
                        "shard": shard_name,
                    }
                )
    if set(weight_map) != {record["name"] for record in records}:
        raise ValueError("snapshot tensor inventory does not match index")
    inventory = build_qwen3_linear_inventory(model_config, records)
    if inventory["included_tensor_count"] != MODEL_CONTRACT["expected_tensor_count"]:
        raise ValueError("included tensor count drifted")
    if inventory["included_parameter_count"] != MODEL_CONTRACT["expected_parameter_count"]:
        raise ValueError("included parameter count drifted")
    return inventory, shard_names


def artifact_stem(tensor_index: int, tensor_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", tensor_name).replace(".", "-")
    return f"{tensor_index:03d}-{safe_name}"


def extended_error_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    difference = target.to(torch.float32) - reconstruction.to(torch.float32)
    squared_error = difference.square().sum(dtype=torch.float64)
    target_squared_norm = target.square().sum(dtype=torch.float64)
    reconstruction_squared_norm = reconstruction.square().sum(dtype=torch.float64)
    dot_product = (target * reconstruction).sum(dtype=torch.float64)
    absolute_error_sum = difference.abs().sum(dtype=torch.float64)
    metrics = {
        "squared_error": float(squared_error),
        "mse": float(squared_error / target.numel()),
        "relative_frobenius_error": float(
            squared_error.sqrt() / target_squared_norm.sqrt()
        ),
        "cosine_similarity": float(
            dot_product / (target_squared_norm * reconstruction_squared_norm).sqrt()
        ),
        "mean_absolute_error": float(absolute_error_sum / target.numel()),
        "max_absolute_error": float(difference.abs().max()),
        "target_squared_norm": float(target_squared_norm),
        "reconstruction_squared_norm": float(reconstruction_squared_norm),
        "dot_product": float(dot_product),
        "absolute_error_sum": float(absolute_error_sum),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("reconstruction metrics contain non-finite values")
    return metrics


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file({name: value.cpu().contiguous() for name, value in tensors.items()}, temporary)
    os.replace(temporary, path)


def write_payload(path: Path, hybrid) -> dict[str, Any]:
    global_value = hybrid.global_decomposition
    refinement = hybrid.refinement_decomposition
    decoded_hashes = {
        "global_signs": tensor_sha256(global_value.bases),
        "refinement_signs": tensor_sha256(refinement.bases),
    }
    tensors = {
        "global_sign_codes": pack_two_bases(global_value.bases),
        "global_row_scales": global_value.row_scales,
        "global_column_scales": global_value.column_scales,
        "refinement_indices": hybrid.selected_indices.to(dtype=torch.int16),
        "refinement_sign_codes": pack_two_bases(refinement.bases),
        "refinement_row_scales": refinement.row_scales,
        "refinement_column_scales": refinement.column_scales,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    atomic_safetensors(path, tensors)
    reopened: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as payload:
        if set(payload.keys()) != set(tensors):
            raise RuntimeError("payload tensor inventory drifted")
        for name in payload.keys():
            reopened[name] = payload.get_tensor(name)
    for name, expected_hash in tensor_hashes.items():
        if tensor_sha256(reopened[name]) != expected_hash:
            raise RuntimeError(f"payload tensor hash drifted: {name}")
    if tensor_sha256(unpack_two_bases(reopened["global_sign_codes"])) != decoded_hashes[
        "global_signs"
    ]:
        raise RuntimeError("global packed-sign round trip failed")
    if tensor_sha256(
        unpack_two_bases(reopened["refinement_sign_codes"])
    ) != decoded_hashes["refinement_signs"]:
        raise RuntimeError("refinement packed-sign round trip failed")
    return {
        "format": TWO_BASE_PACKED_FORMAT,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_sha256": tensor_hashes,
        "decoded_sign_sha256": decoded_hashes,
        "round_trip_verified": True,
        "stored_arms": [
            "global_two_base_rank_one",
            "sparse_residual_refinement_s8",
        ],
        "global_arm_shared_by_pure_and_hybrid": True,
    }


def validate_resumed_metadata(
    metadata: dict[str, Any],
    metadata_path: Path,
    *,
    tensor_index: int,
    tensor_name: str,
    target_hash: str,
    config_hash: str,
    implementation_hash: str,
) -> None:
    expected = {
        "schema_version": 1,
        "status": "passed",
        "tensor_index": tensor_index,
        "tensor_name": tensor_name,
        "target_sha256": target_hash,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "model_revision": MODEL_CONTRACT["revision"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"resumed metadata mismatch for {tensor_name}: {key}")
    payload = metadata["payload"]
    if payload.get("format") != TWO_BASE_PACKED_FORMAT or not payload.get(
        "round_trip_verified"
    ):
        raise ValueError(f"resumed payload contract mismatch for {tensor_name}")
    payload_path = metadata_path.parent / payload["path"]
    if payload_path.stat().st_size != payload["bytes"]:
        raise ValueError(f"resumed payload size mismatch for {tensor_name}")
    if sha256_file(payload_path) != payload["sha256"]:
        raise ValueError(f"resumed payload hash mismatch for {tensor_name}")
    with safe_open(payload_path, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(payload["tensor_sha256"]):
            raise ValueError(f"resumed payload inventory mismatch for {tensor_name}")
        for name, expected_hash in payload["tensor_sha256"].items():
            if tensor_sha256(reopened.get_tensor(name)) != expected_hash:
                raise ValueError(f"resumed tensor hash mismatch for {tensor_name}: {name}")


def record_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "tensor_index": metadata["tensor_index"],
        "tensor_name": metadata["tensor_name"],
        "layer": metadata["layer"],
        "module": metadata["module"],
        "shape": metadata["shape"],
        "parameter_count": metadata["storage"]["weight_count"],
        "metadata_file": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "payload_file": metadata["payload"]["path"],
        "payload_bytes": metadata["payload"]["bytes"],
        "payload_sha256": metadata["payload"]["sha256"],
        "target_sha256": metadata["target_sha256"],
        "global": metadata["reconstruction"]["global_two_base_rank_one"],
        "hybrid": metadata["reconstruction"]["hybrid_s8"],
        "relative_squared_error_reduction": metadata["reconstruction"][
            "relative_squared_error_reduction_vs_global"
        ],
        "selected_residual_energy_fraction": metadata["reconstruction"][
            "selected_residual_energy_fraction"
        ],
        "global_iterations": len(metadata["solver"]["global"]["iterations"]),
        "global_converged_reason": metadata["solver"]["global"]["converged_reason"],
        "refinement_iterations": len(metadata["solver"]["refinement"]["iterations"]),
        "refinement_converged_reason": metadata["solver"]["refinement"][
            "converged_reason"
        ],
        "timing_seconds": metadata["timing_seconds"],
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_count = sum(record["parameter_count"] for record in records)

    def arm_metrics(arm: str) -> dict[str, float]:
        squared_error = sum(record[arm]["squared_error"] for record in records)
        target_norm = sum(record[arm]["target_squared_norm"] for record in records)
        reconstruction_norm = sum(
            record[arm]["reconstruction_squared_norm"] for record in records
        )
        dot = sum(record[arm]["dot_product"] for record in records)
        absolute_error = sum(record[arm]["absolute_error_sum"] for record in records)
        return {
            "squared_error": squared_error,
            "mse": squared_error / parameter_count,
            "relative_frobenius_error": math.sqrt(squared_error / target_norm),
            "cosine_similarity": dot / math.sqrt(target_norm * reconstruction_norm),
            "mean_absolute_error": absolute_error / parameter_count,
            "max_absolute_error": max(record[arm]["max_absolute_error"] for record in records),
            "target_squared_norm": target_norm,
            "reconstruction_squared_norm": reconstruction_norm,
            "dot_product": dot,
            "absolute_error_sum": absolute_error,
        }

    global_metrics = arm_metrics("global")
    hybrid_metrics = arm_metrics("hybrid")
    return {
        "tensor_count": len(records),
        "parameter_count": parameter_count,
        "global_two_base_rank_one": global_metrics,
        "hybrid_s8": hybrid_metrics,
        "squared_error_reduction_vs_global": (
            global_metrics["squared_error"] - hybrid_metrics["squared_error"]
        ),
        "relative_squared_error_reduction_vs_global": (
            global_metrics["squared_error"] - hybrid_metrics["squared_error"]
        )
        / global_metrics["squared_error"],
        "every_tensor_strictly_improved": all(
            record["hybrid"]["squared_error"] < record["global"]["squared_error"]
            for record in records
        ),
    }


def storage_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    weight_count = sum(record["parameter_count"] for record in records)
    global_binary_count = NUM_BASES * weight_count
    refinement_binary_count = 0
    global_row_scale_count = 0
    global_column_scale_count = 0
    refinement_row_scale_count = 0
    refinement_column_scale_count = 0
    refinement_index_count = 0
    for record in records:
        out_features, in_features = record["shape"]
        groups = in_features // ALGORITHM_CONTRACT["global_group_size"]
        selected = ALGORITHM_CONTRACT["residual_columns_per_group"]
        global_row_scale_count += NUM_BASES * out_features * groups
        global_column_scale_count += NUM_BASES * groups * 128
        refinement_binary_count += NUM_BASES * out_features * groups * selected
        refinement_row_scale_count += NUM_BASES * out_features * groups
        refinement_column_scale_count += NUM_BASES * groups * selected
        refinement_index_count += groups * selected
    global_scale_count = global_row_scale_count + global_column_scale_count
    refinement_scale_count = refinement_row_scale_count + refinement_column_scale_count
    global_fp32_bits = global_binary_count + global_scale_count * 32
    hybrid_fp32_bits = (
        global_fp32_bits
        + refinement_binary_count
        + refinement_scale_count * 32
        + refinement_index_count * 16
    )
    global_fp16_bits = global_binary_count + global_scale_count * 16
    hybrid_fp16_bits = (
        global_fp16_bits
        + refinement_binary_count
        + refinement_scale_count * 16
        + refinement_index_count * 16
    )
    return {
        "weight_count": weight_count,
        "global_binary_count": global_binary_count,
        "global_row_scale_count": global_row_scale_count,
        "global_column_scale_count": global_column_scale_count,
        "refinement_binary_count": refinement_binary_count,
        "refinement_row_scale_count": refinement_row_scale_count,
        "refinement_column_scale_count": refinement_column_scale_count,
        "refinement_index_count": refinement_index_count,
        "index_dtype_bits": 16,
        "algorithm_scale_dtype_bits": 32,
        "global_algorithm_effective_bits_per_weight": global_fp32_bits / weight_count,
        "hybrid_algorithm_effective_bits_per_weight": hybrid_fp32_bits / weight_count,
        "global_projected_fp16_scale_effective_bits_per_weight": global_fp16_bits
        / weight_count,
        "hybrid_projected_fp16_scale_effective_bits_per_weight": hybrid_fp16_bits
        / weight_count,
        "projection_is_not_a_runtime_measurement": True,
    }


def main() -> None:
    args = parse_args()
    for required in (args.config, args.single_linear_result, args.source_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config, args.snapshot_root)
    validate_single_linear_gate(args.single_linear_result, config["single_linear_gate"])
    config_hash = sha256_file(args.config)
    implementation_hash = implementation_sha256(args.project_root)
    source_manifest_hash = sha256_file(args.source_manifest)
    model_config_path = args.snapshot_root / "config.json"
    index_path = args.snapshot_root / "model.safetensors.index.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("metadata", {}).get("total_size") != MODEL_CONTRACT[
        "expected_checkpoint_bytes"
    ]:
        raise ValueError("checkpoint byte count drifted")
    inventory, shard_names = snapshot_inventory(args.snapshot_root, model_config, index)
    tensor_names = qwen3_linear_weight_names(MODEL_CONTRACT["expected_hidden_layers"])
    weight_map = index["weight_map"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    execution = config["execution"]
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
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    stale_temporary_count = 0
    for temporary in args.artifact_dir.glob(".*.tmp-*"):
        if temporary.is_file():
            temporary.unlink()
            stale_temporary_count += 1
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    computed_count = 0
    resumed_count = 0
    global_config = TwoBaseRankOneOptimizationConfig(**config["global_solver"])
    refinement_config = TwoBaseRankOneOptimizationConfig(**config["refinement_solver"])

    for tensor_index, tensor_name in enumerate(tensor_names):
        layer, module = parse_qwen3_linear_name(tensor_name)
        shard_path = args.snapshot_root / weight_map[tensor_name]
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            checkpoint_weight = shard.get_tensor(tensor_name).contiguous()
        if str(checkpoint_weight.dtype) != MODEL_CONTRACT["checkpoint_dtype"]:
            raise ValueError(f"checkpoint dtype drifted for {tensor_name}")
        target_hash = tensor_sha256(checkpoint_weight)
        stem = artifact_stem(tensor_index, tensor_name)
        metadata_path = args.artifact_dir / f"{stem}.json"
        payload_path = args.artifact_dir / f"{stem}.safetensors"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_resumed_metadata(
                metadata,
                metadata_path,
                tensor_index=tensor_index,
                tensor_name=tensor_name,
                target_hash=target_hash,
                config_hash=config_hash,
                implementation_hash=implementation_hash,
            )
            resumed_count += 1
            print(
                f"FLUXBIN_FULL_RESUMED={tensor_index + 1}/{len(tensor_names)}:{tensor_name}",
                flush=True,
            )
        else:
            if payload_path.exists():
                # Payload is committed before metadata. Without the metadata commit,
                # it is an incomplete artifact and cannot be trusted or resumed.
                payload_path.unlink()
            target = checkpoint_weight.to(device="cuda:0", dtype=torch.float32)
            tensor_started = time.monotonic()
            global_initial = initialize_two_base_rank_one(target, group_size=128)
            global_optimization = optimize_two_base_rank_one(
                target,
                global_initial,
                global_config,
            )
            global_value = global_optimization.decomposition
            global_reconstruction = global_value.reconstruct()
            global_metrics = extended_error_metrics(target, global_reconstruction)
            refinement = optimize_sparse_residual_refinement(
                target,
                global_value,
                columns_per_group=8,
                config=refinement_config,
            )
            hybrid = refinement.decomposition
            hybrid_reconstruction = hybrid.reconstruct()
            hybrid_metrics = extended_error_metrics(target, hybrid_reconstruction)
            if not hybrid_metrics["squared_error"] < global_metrics["squared_error"]:
                raise RuntimeError(f"hybrid failed strict improvement for {tensor_name}")
            selected_before = gather_grouped_columns(
                target - global_reconstruction,
                hybrid.selected_indices,
                group_size=128,
            ).square().sum(dtype=torch.float64)
            selected_after = gather_grouped_columns(
                target - hybrid_reconstruction,
                hybrid.selected_indices,
                group_size=128,
            ).square().sum(dtype=torch.float64)
            groups = target.shape[1] // 128
            global_grouped = global_reconstruction.reshape(target.shape[0], groups, 128)
            hybrid_grouped = hybrid_reconstruction.reshape_as(global_grouped)
            selected_mask = torch.zeros(groups, 128, dtype=torch.bool, device=target.device)
            selected_mask.scatter_(1, hybrid.selected_indices, True)
            non_selected = (~selected_mask).unsqueeze(0).expand_as(global_grouped)
            non_selected_max_abs_delta = float(
                (hybrid_grouped - global_grouped)[non_selected].abs().max()
            )
            if non_selected_max_abs_delta != 0.0:
                raise RuntimeError(f"non-selected column changed for {tensor_name}")
            if tensor_name == config["single_linear_gate"]["target_tensor"]:
                gate = config["single_linear_gate"]
                tolerance = 1e-6 * gate["global_squared_error"]
                if target_hash != gate["target_tensor_sha256"]:
                    raise RuntimeError("full-model path target differs from single-Linear gate")
                if abs(global_metrics["squared_error"] - gate["global_squared_error"]) > tolerance:
                    raise RuntimeError("full-model global arm does not reproduce single-Linear gate")
                if abs(hybrid_metrics["squared_error"] - gate["hybrid_squared_error"]) > tolerance:
                    raise RuntimeError("full-model hybrid arm does not reproduce single-Linear gate")
            payload = write_payload(payload_path, hybrid)
            weight_count = checkpoint_weight.numel()
            global_sse = global_metrics["squared_error"]
            hybrid_sse = hybrid_metrics["squared_error"]
            metadata = {
                "schema_version": 1,
                "status": "passed",
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "tensor_index": tensor_index,
                "tensor_name": tensor_name,
                "layer": layer,
                "module": module,
                "shape": list(checkpoint_weight.shape),
                "checkpoint_dtype": str(checkpoint_weight.dtype),
                "target_sha256": target_hash,
                "model_revision": MODEL_CONTRACT["revision"],
                "config_sha256": config_hash,
                "implementation_sha256": implementation_hash,
                "source_manifest_path": str(args.source_manifest),
                "source_manifest_sha256": source_manifest_hash,
                "reconstruction": {
                    "global_two_base_rank_one": global_metrics,
                    "hybrid_s8": hybrid_metrics,
                    "squared_error_reduction_vs_global": global_sse - hybrid_sse,
                    "relative_squared_error_reduction_vs_global": (
                        global_sse - hybrid_sse
                    )
                    / global_sse,
                    "selected_residual_energy_fraction": refinement.selected_residual_energy_fraction,
                    "selected_columns_squared_error_before": float(selected_before),
                    "selected_columns_squared_error_after": float(selected_after),
                    "selected_columns_relative_squared_error_reduction": float(
                        (selected_before - selected_after) / selected_before
                    ),
                    "non_selected_max_abs_delta_vs_global": non_selected_max_abs_delta,
                },
                "solver": {
                    "global": {
                        "config": config["global_solver"],
                        "initial_mse": global_optimization.initial_mse,
                        "converged_reason": global_optimization.converged_reason,
                        "iterations": [asdict(item) for item in global_optimization.iterations],
                    },
                    "refinement": {
                        "config": config["refinement_solver"],
                        "initial_mse": refinement.refinement_optimization.initial_mse,
                        "converged_reason": refinement.refinement_optimization.converged_reason,
                        "iterations": [
                            asdict(item)
                            for item in refinement.refinement_optimization.iterations
                        ],
                    },
                },
                "storage": {
                    "weight_count": weight_count,
                    "payload_format": TWO_BASE_PACKED_FORMAT,
                    "global_arm_shared_by_pure_and_hybrid": True,
                },
                "payload": payload,
                "timing_seconds": {
                    "global_optimization": global_optimization.elapsed_seconds,
                    "refinement_optimization": refinement.refinement_optimization.elapsed_seconds,
                    "total_tensor": time.monotonic() - tensor_started,
                },
                "next_stage": "not_launched",
            }
            atomic_json(metadata_path, metadata)
            computed_count += 1
            print(
                f"FLUXBIN_FULL_COMPUTED={tensor_index + 1}/{len(tensor_names)}:{tensor_name}:"
                f"global_sse={global_sse:.9f}:hybrid_sse={hybrid_sse:.9f}:"
                f"relative_reduction={(global_sse-hybrid_sse)/global_sse:.9f}",
                flush=True,
            )
            del target, global_initial, global_optimization, global_value
            del global_reconstruction, refinement, hybrid, hybrid_reconstruction
            torch.cuda.empty_cache()
        records.append(record_from_metadata(metadata_path, metadata))
        del checkpoint_weight, metadata

    aggregate = aggregate_records(records)
    if aggregate["parameter_count"] != MODEL_CONTRACT["expected_parameter_count"]:
        raise RuntimeError("aggregate parameter count drifted")
    if not aggregate["every_tensor_strictly_improved"]:
        raise RuntimeError("at least one tensor failed strict hybrid improvement")
    expected_files = {
        name
        for record in records
        for name in (record["metadata_file"], record["payload_file"])
    }
    observed_files = {path.name for path in args.artifact_dir.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise RuntimeError(
            f"artifact inventory drifted: missing={sorted(expected_files-observed_files)}, "
            f"unexpected={sorted(observed_files-expected_files)}"
        )
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[record["layer"]].append(record)
        by_module[record["module"]].append(record)
    storage = storage_counts(records)
    total_payload_bytes = sum(record["payload_bytes"] for record in records)
    storage.update(
        {
            "payload_format": TWO_BASE_PACKED_FORMAT,
            "stored_arms": config["artifact_policy"]["stored_arms"],
            "global_arm_shared_by_pure_and_hybrid": True,
            "total_payload_bytes": total_payload_bytes,
            "physical_payload_bits_per_weight": total_payload_bytes
            * 8
            / aggregate["parameter_count"],
        }
    )
    global_iterations = [record["global_iterations"] for record in records]
    refinement_iterations = [record["refinement_iterations"] for record in records]
    result = {
        "schema_version": 1,
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "source_manifest_path": str(args.source_manifest),
        "source_manifest_sha256": source_manifest_hash,
        "single_linear_gate": config["single_linear_gate"],
        "model": {
            **MODEL_CONTRACT,
            "snapshot_root": str(args.snapshot_root),
            "config_sha256": sha256_file(model_config_path),
            "index_sha256": sha256_file(index_path),
            "weight_shards": list(shard_names),
            "runtime_inventory": inventory,
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
        "execution": {
            "tensor_count": len(records),
            "computed_count": computed_count,
            "resumed_count": resumed_count,
            "stale_temporary_file_count_removed": stale_temporary_count,
            "artifact_file_count": len(observed_files),
            "elapsed_seconds": time.monotonic() - started,
        },
        "contract": {
            **config["algorithm"],
            "payload_format": TWO_BASE_PACKED_FORMAT,
            "stored_arms": config["artifact_policy"]["stored_arms"],
            "global_arm_shared_by_pure_and_hybrid": True,
            "one_payload_per_tensor": True,
            "resumable": True,
        },
        "aggregate": aggregate,
        "solver_summary": {
            "global": {
                "min": min(global_iterations),
                "median": statistics.median(global_iterations),
                "max": max(global_iterations),
                "mean": statistics.fmean(global_iterations),
                "converged_reasons": dict(
                    sorted(Counter(record["global_converged_reason"] for record in records).items())
                ),
            },
            "refinement": {
                "min": min(refinement_iterations),
                "median": statistics.median(refinement_iterations),
                "max": max(refinement_iterations),
                "mean": statistics.fmean(refinement_iterations),
                "converged_reasons": dict(
                    sorted(
                        Counter(
                            record["refinement_converged_reason"] for record in records
                        ).items()
                    )
                ),
            },
        },
        "storage": storage,
        "by_layer": {
            str(layer): aggregate_records(values) for layer, values in sorted(by_layer.items())
        },
        "by_module": {
            module: aggregate_records(values) for module, values in sorted(by_module.items())
        },
        "tensors": records,
        "decision": config["decision"],
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_FULL_RESULT={args.output}")
    print(f"FLUXBIN_FULL_ARTIFACT_DIR={args.artifact_dir}")
    print(
        "FLUXBIN_FULL_EFFECT="
        f"global_sse={aggregate['global_two_base_rank_one']['squared_error']:.9f} "
        f"hybrid_sse={aggregate['hybrid_s8']['squared_error']:.9f} "
        f"relative_reduction={aggregate['relative_squared_error_reduction_vs_global']:.9f}",
        flush=True,
    )
    print("FLUXBIN_FULL_PASSED")


if __name__ == "__main__":
    main()
