#!/usr/bin/env python3
"""Create the deployment-realistic QBB FP16-scale arm without refitting."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import QWEN3_LINEAR_MODULES, atomic_json, sha256_file, tensor_sha256
from fluxbin_style.evaluation import materialize_hybrid_s8_weight
from fluxbin_style.rate_distortion import (
    EXPECTED_LAYERS,
    EXPECTED_LINEARS,
    EXPECTED_WEIGHTS,
    FIXED_QBB_SUFFIXES,
    SCALE_SUFFIXES,
    analytical_qbb_storage,
    summarize_tensor_storage,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source-acceptance", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-payload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def expected_keys() -> set[str]:
    return {
        f"{module}.{suffix}"
        for module in QWEN3_LINEAR_MODULES
        for suffix in (*FIXED_QBB_SUFFIXES, *SCALE_SUFFIXES)
    }


def validate_source(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    qbb = config["qbb"]
    expected_contract = {
        "artifact_id": "qwen3-8b-hybrid-distilled-step400-v1",
        "payload_format": "fluxbin-two-base-interleaved-2bit-v1",
        "group_size": 128,
        "columns_per_group": 8,
        "current_scale_dtype": "torch.float32",
        "deployment_scale_dtype": "torch.float16",
        "cast_scale_suffixes": list(SCALE_SUFFIXES),
        "fixed_suffixes": list(FIXED_QBB_SUFFIXES),
        "include_legacy_lookup_in_deployment_rate": False,
    }
    observed_contract = {name: qbb.get(name) for name in expected_contract}
    if observed_contract != expected_contract:
        raise ValueError(f"frozen QBB FP16-scale contract drifted: {observed_contract}")
    if sha256_file(args.source_acceptance) != qbb["accepted_acceptance_sha256"]:
        raise ValueError("accepted QBB review hash drifted")
    if sha256_file(args.source_result) != qbb["accepted_result_sha256"]:
        raise ValueError("accepted QBB result hash drifted")
    acceptance = json.loads(args.source_acceptance.read_text(encoding="utf-8"))
    result = json.loads(args.source_result.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted_validation_only":
        raise ValueError("QBB source is not accepted")
    if acceptance.get("result_sha256") != qbb["accepted_result_sha256"]:
        raise ValueError("QBB review does not bind the accepted result")
    payloads = result.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != EXPECTED_LAYERS:
        raise ValueError("QBB source does not contain 36 layer payloads")
    if [entry.get("layer") for entry in payloads] != list(range(EXPECTED_LAYERS)):
        raise ValueError("QBB source layer order drifted")
    for entry in payloads:
        path = args.source_payload_dir / f"layer-{entry['layer']:03d}.safetensors"
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"QBB layer payload hash drifted: {entry['layer']}")
    return payloads


def convert_tensors(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(tensors) != expected_keys():
        raise ValueError("QBB tensor inventory drifted")
    converted: dict[str, torch.Tensor] = {}
    fixed_hashes: dict[str, str] = {}
    scale_hashes: dict[str, dict[str, str]] = {}
    for name, value in tensors.items():
        suffix = name.rsplit(".", 1)[1]
        if suffix in SCALE_SUFFIXES:
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"expected finite FP32 source scale: {name}")
            cast = value.to(torch.float16)
            if not torch.isfinite(cast).all():
                raise ValueError(f"FP16 scale overflowed: {name}")
            converted[name] = cast
            scale_hashes[name] = {
                "source_fp32": tensor_sha256(value),
                "derived_fp16": tensor_sha256(cast),
            }
        elif suffix in FIXED_QBB_SUFFIXES:
            converted[name] = value.clone()
            fixed_hashes[name] = tensor_sha256(value)
        else:
            raise AssertionError(name)
    return converted, {"fixed_tensor_sha256": fixed_hashes, "scale_tensor_sha256": scale_hashes}


def validate_decoded_layer(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for module in QWEN3_LINEAR_MODULES:
        prefix = f"{module}."
        weight = materialize_hybrid_s8_weight(
            tensors[prefix + "global_sign_codes"],
            tensors[prefix + "global_row_scales"],
            tensors[prefix + "global_column_scales"],
            tensors[prefix + "refinement_indices"],
            tensors[prefix + "refinement_sign_codes"],
            tensors[prefix + "refinement_row_scales"],
            tensors[prefix + "refinement_column_scales"],
            group_size=128,
            columns_per_group=8,
            device="cpu",
            output_dtype=torch.bfloat16,
        )
        if weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
            raise ValueError(f"invalid BF16 reconstruction: {module}")
        hashes[module] = tensor_sha256(weight)
    return hashes


def convert_layer(source: Path, destination: Path, *, layer: int) -> dict[str, Any]:
    with safe_open(source, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    converted, hashes = convert_tensors(tensors)
    decoded_hashes = validate_decoded_layer(converted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".layer-{layer:03d}-", dir=destination.parent))
    payload_path = temporary / "payload.safetensors"
    save_file({name: value.contiguous() for name, value in converted.items()}, payload_path)
    with safe_open(payload_path, framework="pt", device="cpu") as reopened:
        reopened_tensors = {name: reopened.get_tensor(name) for name in reopened.keys()}
    if set(reopened_tensors) != set(converted):
        raise RuntimeError("derived QBB payload inventory changed after serialization")
    fixed_hashes = hashes["fixed_tensor_sha256"]
    for name in fixed_hashes:
        if tensor_sha256(reopened_tensors[name]) != fixed_hashes[name]:
            raise RuntimeError(f"fixed QBB tensor changed during conversion: {name}")
    for name, record in hashes["scale_tensor_sha256"].items():
        if tensor_sha256(reopened_tensors[name]) != record["derived_fp16"]:
            raise RuntimeError(f"FP16 QBB scale changed during serialization: {name}")
    storage = summarize_tensor_storage(reopened_tensors)
    if storage["uncategorized"]:
        raise RuntimeError("derived QBB payload has uncategorized tensors")
    metadata = {
        "schema_version": 1,
        "status": "passed",
        "layer": layer,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "payload_path": "payload.safetensors",
        "payload_sha256": sha256_file(payload_path),
        "payload_bytes": payload_path.stat().st_size,
        "tensor_storage": storage,
        "decoded_bf16_sha256": decoded_hashes,
        **hashes,
    }
    atomic_json(temporary / "metadata.json", metadata)
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(temporary, destination)
    return metadata


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payloads = validate_source(config, args)
    if args.validate_only:
        print("FLUXBIN_QBB_FP16_PREFLIGHT=passed; no conversion launched")
        return
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in payloads:
        layer = int(entry["layer"])
        source = args.source_payload_dir / f"layer-{layer:03d}.safetensors"
        destination = args.output_dir / "payloads" / f"layer-{layer:03d}"
        if destination.exists():
            metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("source_sha256") != entry["sha256"]:
                raise ValueError(f"resumed derived layer source drifted: {layer}")
            if sha256_file(destination / "payload.safetensors") != metadata["payload_sha256"]:
                raise ValueError(f"resumed derived layer payload drifted: {layer}")
        else:
            metadata = convert_layer(source, destination, layer=layer)
        records.append(metadata)
        print(f"FLUXBIN_QBB_FP16_LAYER={layer + 1}/{EXPECTED_LAYERS}", flush=True)
    fixed_unchanged = all(
        len(record["fixed_tensor_sha256"]) == len(QWEN3_LINEAR_MODULES) * len(FIXED_QBB_SUFFIXES)
        for record in records
    )
    scale_count = sum(len(record["scale_tensor_sha256"]) for record in records)
    if not fixed_unchanged or scale_count != EXPECTED_LINEARS * len(SCALE_SUFFIXES):
        raise RuntimeError("derived QBB coverage drifted")
    analytical = analytical_qbb_storage(config["model"], scale_bytes=2, include_lookup=False)
    tensor_bytes = sum(record["tensor_storage"]["tensor_bytes"] for record in records)
    if tensor_bytes != analytical["persistent_tensor_bytes"]:
        raise RuntimeError("derived QBB tensor bytes disagree with analytical accounting")
    result = {
        "schema_version": 1,
        "status": "completed_pending_review",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": "qbb_fp16_scales",
        "source_artifact_id": config["qbb"]["artifact_id"],
        "source_result_sha256": sha256_file(args.source_result),
        "config_sha256": sha256_file(args.config),
        "layer_count": EXPECTED_LAYERS,
        "linear_count": EXPECTED_LINEARS,
        "quantized_weight_count": EXPECTED_WEIGHTS,
        "fixed_binary_and_index_hashes_unchanged": fixed_unchanged,
        "scale_tensor_count": scale_count,
        "scale_conversion": "FP32 to FP16 only; no optimization or training",
        "analytical_storage": analytical,
        "serialized_tensor_bytes": tensor_bytes,
        "serialized_file_bytes": sum(record["payload_bytes"] for record in records),
        "layers": [
            {
                "layer": record["layer"],
                "payload": f"payloads/layer-{record['layer']:03d}/payload.safetensors",
                "payload_sha256": record["payload_sha256"],
                "metadata": f"payloads/layer-{record['layer']:03d}/metadata.json",
                "metadata_sha256": sha256_file(args.output_dir / "payloads" / f"layer-{record['layer']:03d}" / "metadata.json"),
            }
            for record in records
        ],
    }
    atomic_json(result_path, result)
    print(f"FLUXBIN_QBB_FP16_RESULT={result_path}")


if __name__ == "__main__":
    main()
