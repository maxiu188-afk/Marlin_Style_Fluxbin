#!/usr/bin/env python3
"""Inspect all 252 raw GPTQ W3 Linears and create exact planar LUT payloads.

This is an offline CPU conversion. It does not compile CUDA or run a benchmark.
The accepted raw FORMAT.GPTQ checkpoint is decoded with explicit qzero format 1,
and every converted weight must equal the retained GPTQModel BF16 decoder output
bit-for-bit before any deployment payload is published.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256
from fluxbin_style.gptq_deployment import (
    FIELDS,
    FORMAT,
    convert_gptq_w3_to_planar,
    decode_planar_w3_codes,
    inspect_gptq_checkpoint,
    layout_storage,
    restore_planar_w3,
    unpack_gptq_w3_qweight,
)
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES


EXPECTED_SOURCE_MANIFEST_SHA256 = "96b59535af55f30fe4b3992baa565800bfaaf659975d3e5fb501af0bfc17e482"
EXPECTED_LAYERS = 36
EXPECTED_LINEARS = 252
REQUIRED_RAW = ("qweight", "qzeros", "scales", "g_idx")
W3_CONTRACT_FILES = (
    "configs/acceleration/w3_lut_candidates_v1.json",
    "scripts/record_acceleration_environment.py",
    "scripts/prepare_qwen3_8b_w3_lut_artifacts.py",
    "scripts/run_w3_lut_server_preflight.py",
    "scripts/run_w3_lut_benchmark.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--inspection-output",
        type=Path,
        help="optional JSON record for the validate-only canonical-semantics pass",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256", default=EXPECTED_SOURCE_MANIFEST_SHA256
    )
    args = parser.parse_args()
    if args.validate_only == (args.output_dir is not None):
        parser.error("use exactly one of --validate-only or --output-dir")
    if args.inspection_output is not None and not args.validate_only:
        parser.error("--inspection-output is valid only with --validate-only")
    return args


def safetensor_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate safetensor key: {key}")
                result[key] = path
    if not result:
        raise FileNotFoundError(f"no safetensors under {root}")
    return result


def load_tensor(index: dict[str, Path], key: str) -> torch.Tensor:
    path = index.get(key)
    if path is None:
        raise KeyError(key)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def validate_source(root: Path, expected_sha: str) -> tuple[dict, dict[str, Path]]:
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_sha:
        raise ValueError("accepted GPTQ source manifest hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_pending_review":
        raise ValueError("GPTQ source status drifted")
    # Require the four deployment-defining values without freezing unrelated
    # calibration fields a second time.
    for name, expected in (("bits", 3), ("group_size", 128), ("sym", True), ("desc_act", True)):
        if manifest.get("gptq", {}).get(name) != expected:
            raise ValueError(f"GPTQ source config drifted: {name}")
    if manifest.get("coverage", {}).get("linear_count") != EXPECTED_LINEARS:
        raise ValueError("GPTQ source coverage drifted")
    raw_root = root / "raw_quantized"
    decoded_root = root / "decoded_bf16"
    if not raw_root.is_dir() or not decoded_root.is_dir():
        raise FileNotFoundError("raw_quantized/decoded_bf16 source directories are required")
    index = safetensor_index(raw_root)
    targets = {
        f"model.layers.{layer}.{module}"
        for layer in range(EXPECTED_LAYERS)
        for module in QWEN3_LINEAR_MODULES
    }
    observed = {key.rsplit(".", 1)[0] for key in index if key.rsplit(".", 1)[-1] in REQUIRED_RAW}
    if observed != targets:
        raise ValueError(f"raw GPTQ target coverage drifted: missing={len(targets-observed)} extra={len(observed-targets)}")
    for target in targets:
        if any(f"{target}.{field}" not in index for field in REQUIRED_RAW):
            raise ValueError(f"incomplete raw GPTQ module: {target}")
    for layer in range(EXPECTED_LAYERS):
        if not (decoded_root / f"layer-{layer:03d}.safetensors").is_file():
            raise FileNotFoundError(f"missing decoded layer {layer}")
    return manifest, index


def inspect_all_modules(index: dict[str, Path]) -> tuple[list[dict], int]:
    """Complete the canonical-semantics gate before any normalization writes."""
    records = []
    nonidentity = 0
    for layer in range(EXPECTED_LAYERS):
        for module in QWEN3_LINEAR_MODULES:
            full_name = f"model.layers.{layer}.{module}"
            raw = {field: load_tensor(index, f"{full_name}.{field}") for field in REQUIRED_RAW}
            facts = inspect_gptq_checkpoint(raw, qzero_format=1)
            if facts["decoded_zero_unique"] != [4]:
                raise ValueError(f"non-constant decoded zero: {full_name}")
            records.append({"name": full_name, "facts": facts})
            nonidentity += int(not facts["permutation_identity"])
            del raw
    if len(records) != EXPECTED_LINEARS:
        raise RuntimeError("canonical-semantics inspection coverage drifted")
    return records, nonidentity


def main():
    args = parse_args()
    source_manifest, index = validate_source(
        args.gptq_root, args.expected_source_manifest_sha256
    )
    inspection_records, inspected_nonidentity = inspect_all_modules(index)
    if args.validate_only:
        inspection = {
            "schema_version": 1,
            "status": "canonical_semantics_confirmed",
            "source_manifest_sha256": args.expected_source_manifest_sha256,
            "gptqmodel_version": "7.4.0",
            "raw_qzero_format": 1,
            "desc_act": True,
            "canonicalization": "per-Linear stable argsort(g_idx); same permutation for weight K and runtime x",
            "coverage": {
                "layers": EXPECTED_LAYERS,
                "linears": len(inspection_records),
                "nonidentity_permutations": inspected_nonidentity,
                "all_groups_have_128_columns": True,
                "decoded_zero_unique": [4],
            },
            "modules": inspection_records,
        }
        if args.inspection_output is not None:
            if args.inspection_output.exists():
                raise FileExistsError(args.inspection_output)
            atomic_json(args.inspection_output, inspection)
        print("FLUXBIN_W3_LUT_SOURCE_PREFLIGHT=passed; no conversion launched")
        return

    staging = args.output_dir.with_name(args.output_dir.name + ".incomplete")
    if args.output_dir.exists() or staging.exists():
        raise FileExistsError(args.output_dir if args.output_dir.exists() else staging)
    (staging / "payloads").mkdir(parents=True)
    files = []
    module_records = []
    nonidentity = 0
    total_payload_bytes = 0
    try:
        for layer in range(EXPECTED_LAYERS):
            decoded_path = args.gptq_root / "decoded_bf16" / f"layer-{layer:03d}.safetensors"
            with safe_open(decoded_path, framework="pt", device="cpu") as decoded_handle:
                expected_decoded = {f"{module}.weight" for module in QWEN3_LINEAR_MODULES}
                if set(decoded_handle.keys()) != expected_decoded:
                    raise ValueError(f"decoded layer inventory drifted: {layer}")
                output_tensors = {}
                for module in QWEN3_LINEAR_MODULES:
                    full_name = f"model.layers.{layer}.{module}"
                    raw = {field: load_tensor(index, f"{full_name}.{field}") for field in REQUIRED_RAW}
                    facts = inspect_gptq_checkpoint(raw, qzero_format=1)
                    if facts["decoded_zero_unique"] != [4]:
                        raise ValueError(f"non-constant decoded zero: {full_name}")
                    layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
                    if not torch.equal(layout["scales"], raw["scales"].to(torch.bfloat16)):
                        raise RuntimeError(f"loaded-BF16 scale cast failed: {full_name}")
                    codes = unpack_gptq_w3_qweight(raw["qweight"])
                    if not torch.equal(decode_planar_w3_codes(layout), codes):
                        raise RuntimeError(f"integer code round-trip failed: {full_name}")
                    restored = restore_planar_w3(layout, dtype=torch.bfloat16)
                    decoded = decoded_handle.get_tensor(f"{module}.weight").contiguous()
                    if not torch.equal(restored, decoded):
                        mismatch = int(torch.count_nonzero(restored != decoded))
                        raise RuntimeError(f"BF16 bitwise round-trip failed: {full_name}; mismatch={mismatch}")
                    storage = layout_storage(layout)
                    total_payload_bytes += int(storage["total_bytes"])
                    nonidentity += int(not facts["permutation_identity"])
                    module_records.append(
                        {
                            "name": full_name,
                            "facts": facts,
                            "storage": storage,
                            "decoded_bf16_sha256": tensor_sha256(decoded),
                            "source_fp16_scales_sha256": tensor_sha256(raw["scales"]),
                            "deployment_bf16_scales_sha256": tensor_sha256(layout["scales"]),
                            "integer_codes_exact": True,
                            "decoded_zero_exact": True,
                            "deployment_scales_match_loaded_bf16_cast": True,
                            "decoded_bf16_bitwise_exact": True,
                        }
                    )
                    for field in FIELDS:
                        output_tensors[f"{module}.{field}"] = layout[field]
                    del raw, layout, codes, restored, decoded

            relative = Path("payloads") / f"layer-{layer:03d}.safetensors"
            path = staging / relative
            save_file(output_tensors, path)
            files.append(
                {
                    "layer": layer,
                    "path": str(relative),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            del output_tensors
            gc.collect()
            print(f"FLUXBIN_W3_LUT_LAYER={layer + 1}/{EXPECTED_LAYERS}", flush=True)

        if len(module_records) != EXPECTED_LINEARS:
            raise RuntimeError("converted Linear coverage drifted")
        source_files = sorted((Path(__file__).resolve().parents[1] / "src/fluxbin_style").rglob("*"))
        result = {
            "schema_version": 1,
            "status": "completed_pending_gpu_validation",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "format": FORMAT,
            "source_manifest_sha256": args.expected_source_manifest_sha256,
            "source_runtime": source_manifest.get("runtime"),
            "qzero_format": 1,
            "canonicalization": "per-Linear stable argsort(g_idx); identical permutation applied to K weights and runtime x",
            "constant_zero_fast_path": 4,
            "canonical_semantics_gate": {
                "status": "confirmed_before_payload_writes",
                "inspected_linears": len(inspection_records),
                "nonidentity_permutations": inspected_nonidentity,
                "all_groups_have_128_columns": True,
                "decoded_zero_unique": [4],
            },
            "coverage": {
                "layers": EXPECTED_LAYERS,
                "linears": len(module_records),
                "nonidentity_permutations": nonidentity,
                "integer_code_round_trip_exact": True,
                "decoded_zero_exact": True,
                "deployment_scales_match_loaded_bf16_cast": True,
                "decoded_bf16_bitwise_exact": True,
            },
            "deployment_tensor_bytes": total_payload_bytes,
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
                for path in source_files
                if path.suffix in {".py", ".cu", ".cuh"}
            },
            "w3_contract_sha256": {
                name: sha256_file(Path(__file__).resolve().parents[1] / name)
                for name in W3_CONTRACT_FILES
            },
            "files": files,
            "modules": module_records,
        }
        atomic_json(staging / "manifest.json", result)
        staging.rename(args.output_dir)
    except Exception:
        # Preserve a failed staging tree for diagnosis; never publish it under
        # the requested final artifact path.
        if staging.exists():
            atomic_json(staging / "failure.json", {"status": "failed"})
        raise
    print(f"FLUXBIN_W3_LUT_ARTIFACT={args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
