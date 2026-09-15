#!/usr/bin/env python3
"""Recover a standard GPTQ checkpoint from a completed disk-offload run.

This is intentionally a recovery-only path.  It never quantizes weights.  It
combines the pinned source checkpoint's pass-through tensors with the 252
already packed GPTQ module bundles, then exercises the normal GPTQModel reload
and dense-BF16 decode gates before writing the experiment manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import atomic_json, sha256_file
from fluxbin_style.rate_distortion import (
    EXPECTED_LAYERS,
    EXPECTED_LINEARS,
    EXPECTED_WEIGHTS,
    analytical_gptq_storage,
)

from run_qwen3_8b_w3_gptq import (
    CONFIG,
    audit_raw_artifact,
    decode_artifact,
    target_modules,
    validate_inputs,
    validate_runtime,
)


RECOVERY_REASON = (
    "GPTQModel 7.4.0 completed all 252 disk-offloaded quantized modules, then "
    "rejected split_by='layer' because that release temporarily supports only None"
)
PASSTHROUGH_FILES = (
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failed-job-dir", type=Path)
    return parser.parse_args()


def inspect_offload(offload_dir: Path) -> dict[str, Any]:
    expected = set(target_modules())
    observed: dict[str, dict[str, Any]] = {}
    tensor_bytes: dict[str, int] = {}
    for module_name in sorted(expected):
        path = offload_dir / module_name / "module.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"missing offloaded GPTQ module: {module_name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            fields = set(handle.keys())
            if fields != {"qweight", "qzeros", "scales", "g_idx"}:
                raise ValueError(f"offloaded GPTQ fields drifted: {module_name}={sorted(fields)}")
            field_records = {}
            for field in sorted(fields):
                tensor = handle.get_tensor(field)
                if field == "scales":
                    if tensor.dtype not in {torch.float16, torch.bfloat16}:
                        raise ValueError(f"offloaded scale dtype drifted: {module_name}={tensor.dtype}")
                elif tensor.dtype != torch.int32:
                    raise ValueError(f"offloaded packed dtype drifted: {module_name}.{field}={tensor.dtype}")
                size = tensor.numel() * tensor.element_size()
                tensor_bytes[field] = tensor_bytes.get(field, 0) + size
                field_records[field] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": size,
                }
        observed[module_name] = {
            "path": str(path.relative_to(offload_dir.parent)),
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "tensors": field_records,
        }
    unexpected = sorted(
        str(path.parent.relative_to(offload_dir))
        for path in offload_dir.glob("model.layers.*/module.safetensors")
        if str(path.parent.relative_to(offload_dir)) not in expected
    )
    if unexpected:
        raise ValueError(f"unexpected offloaded layer modules: {unexpected}")
    return {
        "quantized_module_count": len(observed),
        "tensor_bytes": dict(sorted(tensor_bytes.items())),
        "quantized_tensor_bytes": sum(tensor_bytes.values()),
        "modules": observed,
    }


def copy_checkpoint_support_files(source_dir: Path, stub_dir: Path, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True)
    for name in ("config.json", "generation_config.json", "quantize_config.json", "quant_log.csv"):
        source = stub_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"failed save stub is missing {name}")
        shutil.copy2(source, staging_dir / name)
    for name in PASSTHROUGH_FILES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"source snapshot is missing {name}")
        shutil.copy2(source, staging_dir / name)


def merge_checkpoint(source_dir: Path, offload_dir: Path, staging_dir: Path) -> dict[str, Any]:
    source_index_path = source_dir / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict):
        raise ValueError("source checkpoint index has no weight_map")

    targets = set(target_modules())
    target_weights = {f"{name}.weight" for name in targets}
    if not target_weights.issubset(source_weight_map):
        missing = sorted(target_weights - set(source_weight_map))
        raise ValueError(f"source checkpoint is missing target weights: {missing}")

    shard_names = sorted(set(source_weight_map.values()))
    output_weight_map: dict[str, str] = {}
    output_files = []
    replaced: set[str] = set()
    total_tensor_bytes = 0
    metadata = {
        "format": "pt",
        "recovered_from": "gptqmodel-disk-offload",
    }

    for shard_index, shard_name in enumerate(shard_names, start=1):
        source_path = source_dir / shard_name
        output_path = staging_dir / shard_name
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source_path, framework="pt", device="cpu") as source:
            for tensor_name in source.keys():
                if tensor_name in target_weights:
                    module_name = tensor_name.removesuffix(".weight")
                    offload_path = offload_dir / module_name / "module.safetensors"
                    with safe_open(offload_path, framework="pt", device="cpu") as packed:
                        for field in packed.keys():
                            recovered_name = f"{module_name}.{field}"
                            tensors[recovered_name] = packed.get_tensor(field)
                            output_weight_map[recovered_name] = shard_name
                    replaced.add(module_name)
                else:
                    tensors[tensor_name] = source.get_tensor(tensor_name)
                    output_weight_map[tensor_name] = shard_name
        save_file(tensors, temp_path, metadata=metadata)
        os.replace(temp_path, output_path)
        shard_tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        total_tensor_bytes += shard_tensor_bytes
        output_files.append(
            {
                "path": shard_name,
                "bytes": output_path.stat().st_size,
                "tensor_bytes": shard_tensor_bytes,
                "sha256": sha256_file(output_path),
            }
        )
        print(f"FLUXBIN_GPTQ_RECOVERED_SHARD={shard_index}/{len(shard_names)}", flush=True)

    if replaced != targets:
        missing = sorted(targets - replaced)
        raise ValueError(f"recovery did not replace every target Linear: {missing}")
    index = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    atomic_json(staging_dir / "model.safetensors.index.json", index)
    return {
        "source_shard_count": len(shard_names),
        "replaced_linear_count": len(replaced),
        "whole_model_tensor_bytes": total_tensor_bytes,
        "files": output_files,
        "index_sha256": sha256_file(staging_dir / "model.safetensors.index.json"),
    }


def inspect_failed_job(job_dir: Path | None) -> dict[str, Any] | None:
    if job_dir is None:
        return None
    records = {}
    for name in ("run.log", "exit-code", "environment-freeze.txt", "git-revision"):
        path = job_dir / name
        if path.is_file():
            records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    exit_code_path = job_dir / "exit-code"
    if not exit_code_path.is_file() or exit_code_path.read_text(encoding="utf-8").strip() != "1":
        raise ValueError("recovery must be bound to the expected failed save attempt")
    return {"path": str(job_dir), "files": records}


def promote_recovered_raw(staging_dir: Path, stub_dir: Path, preserved_stub: Path) -> None:
    if preserved_stub.exists():
        raise FileExistsError(preserved_stub)
    stub_dir.rename(preserved_stub)
    staging_dir.rename(stub_dir)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _tokens = validate_inputs(config, args)
    validate_runtime(config)
    del _tokens

    output_dir = args.output_dir
    stub_dir = output_dir / "raw_quantized"
    offload_dir = output_dir / "offload"
    staging_dir = output_dir / "raw_quantized.recovering"
    preserved_stub = output_dir / "failed_save_stub"
    decoded_dir = output_dir / "decoded_bf16"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or staging_dir.exists() or preserved_stub.exists() or decoded_dir.exists():
        raise FileExistsError("recovery destination already contains promoted or partial recovery outputs")
    if not stub_dir.is_dir() or not offload_dir.is_dir():
        raise FileNotFoundError("failed save stub or GPTQ offload directory is missing")

    offload_audit = inspect_offload(offload_dir)
    if offload_audit["quantized_module_count"] != EXPECTED_LINEARS:
        raise ValueError("offload does not cover exactly 252 quantized Linears")
    failed_job = inspect_failed_job(args.failed_job_dir)
    copy_checkpoint_support_files(args.snapshot_root, stub_dir, staging_dir)
    merged = merge_checkpoint(args.snapshot_root, offload_dir, staging_dir)
    raw_audit = audit_raw_artifact(staging_dir)
    decoded = decode_artifact(config, staging_dir, decoded_dir)
    decoded_linears = sum(len(layer["tensors"]) for layer in decoded)
    decoded_weights = sum(item["numel"] for layer in decoded for item in layer["tensors"])
    if len(decoded) != EXPECTED_LAYERS or decoded_linears != EXPECTED_LINEARS or decoded_weights != EXPECTED_WEIGHTS:
        raise RuntimeError("recovered GPTQ decoded BF16 coverage drifted")

    promote_recovered_raw(staging_dir, stub_dir, preserved_stub)
    manifest = {
        "schema_version": 1,
        "status": "completed_pending_review",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "model": config["model"],
        "calibration": config["calibration"],
        "gptq": config["gptq"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gptqmodel": importlib.metadata.version("gptqmodel"),
            "device": torch.cuda.get_device_name(0),
            "gptq_torch_triton_dequant": os.environ.get("GPTQ_TORCH_TRITON_DEQUANT", "0"),
        },
        "coverage": {
            "layer_count": len(decoded),
            "linear_count": decoded_linears,
            "quantized_weight_count": decoded_weights,
            "no_fp_fallback": True,
            "decoded_dtype": "torch.bfloat16",
            "decoded_all_finite": True,
        },
        "analytical_storage": analytical_gptq_storage(config["model"]),
        "raw_artifact": raw_audit,
        "decoded_layers": decoded,
        "quantization_log": {
            "source": "failed_save_stub/quant_log.csv",
            "sha256": sha256_file(preserved_stub / "quant_log.csv"),
        },
        "recovery": {
            "reason": RECOVERY_REASON,
            "method": "replace 252 pinned source Linear weights with completed GPTQ disk-offload bundles",
            "offload": offload_audit,
            "merged_checkpoint": merged,
            "failed_job": failed_job,
            "post_recovery_standard_reload": True,
        },
    }
    atomic_json(manifest_path, manifest)
    print(f"FLUXBIN_GPTQ_W3_RECOVERED={manifest_path}")


if __name__ == "__main__":
    main()
