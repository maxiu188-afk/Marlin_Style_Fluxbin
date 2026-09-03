#!/usr/bin/env python3
"""Compare bounded full-model artifacts against retained layer oracles."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fluxbin_style import atomic_json, sha256_file, tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("pure", "hybrid_s8"), required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--last-layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_equal(name: str, reference: Any, candidate: Any) -> None:
    if reference != candidate:
        raise ValueError(f"artifact comparison failed for {name}")


def validate_stage_timings(metadata: dict[str, Any], layer_index: int) -> None:
    stage = metadata.get("stage_elapsed_seconds")
    if not isinstance(stage, dict):
        raise ValueError(f"candidate layer {layer_index} lacks stage timings")
    expected = {"hessian_capture", "hessian_inversion", "linear_quantization"}
    if set(stage) != expected:
        raise ValueError(f"candidate layer {layer_index} stage timing inventory drifted")
    values = [stage["hessian_capture"]]
    values.extend(stage["hessian_inversion"].values())
    values.extend(stage["linear_quantization"].values())
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"candidate layer {layer_index} stage timing is invalid")


def compare_payloads(
    reference_path: Path,
    candidate_path: Path,
    *,
    layer_index: int,
) -> tuple[str, int, int]:
    reference_hash = sha256_file(reference_path)
    candidate_hash = sha256_file(candidate_path)
    require_equal(f"layer {layer_index} payload SHA-256", reference_hash, candidate_hash)
    with safe_open(reference_path, framework="pt", device="cpu") as reference, safe_open(
        candidate_path,
        framework="pt",
        device="cpu",
    ) as candidate:
        reference_keys = sorted(reference.keys())
        candidate_keys = sorted(candidate.keys())
        require_equal(
            f"layer {layer_index} payload inventory",
            reference_keys,
            candidate_keys,
        )
        for name in reference_keys:
            left = reference.get_tensor(name)
            right = candidate.get_tensor(name)
            require_equal(f"layer {layer_index}:{name} dtype", left.dtype, right.dtype)
            require_equal(
                f"layer {layer_index}:{name} shape",
                tuple(left.shape),
                tuple(right.shape),
            )
            if not torch.equal(left, right):
                raise ValueError(f"layer {layer_index}:{name} tensor values drifted")
            require_equal(
                f"layer {layer_index}:{name} tensor SHA-256",
                tensor_sha256(left),
                tensor_sha256(right),
            )
    return reference_hash, reference_path.stat().st_size, len(reference_keys)


def main() -> None:
    args = parse_args()
    if args.last_layer < 0:
        raise ValueError("last_layer must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    layers: list[dict[str, Any]] = []
    candidate_config_hashes: set[str] = set()
    candidate_implementation_hashes: set[str] = set()
    total_payload_bytes = 0
    total_payload_tensors = 0
    for layer_index in range(args.last_layer + 1):
        reference_dir = args.reference_root / f"layer-{layer_index:03d}"
        candidate_dir = args.candidate_root / f"layer-{layer_index:03d}"
        reference_metadata_path = reference_dir / "metadata.json"
        candidate_metadata_path = candidate_dir / "metadata.json"
        if not reference_metadata_path.is_file() or not candidate_metadata_path.is_file():
            raise FileNotFoundError(f"missing layer metadata: {layer_index}")
        reference = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
        for name in (
            "schema_version",
            "status",
            "arm",
            "layer_index",
            "model_revision",
            "calibration_manifest_sha256",
            "hessians",
            "linears",
        ):
            require_equal(
                f"layer {layer_index} metadata {name}",
                reference[name],
                candidate[name],
            )
        require_equal(f"layer {layer_index} arm", candidate["arm"], args.arm)
        validate_stage_timings(candidate, layer_index)
        candidate_config_hashes.add(candidate["config_sha256"])
        candidate_implementation_hashes.add(candidate["implementation_sha256"])
        payload_hash, payload_bytes, payload_tensors = compare_payloads(
            reference_dir / "payload.safetensors",
            candidate_dir / "payload.safetensors",
            layer_index=layer_index,
        )
        require_equal(
            f"layer {layer_index} reference metadata payload hash",
            reference["payload"]["sha256"],
            payload_hash,
        )
        require_equal(
            f"layer {layer_index} candidate metadata payload hash",
            candidate["payload"]["sha256"],
            payload_hash,
        )
        require_equal(
            f"layer {layer_index} payload tensor hashes",
            reference["payload"]["tensor_sha256"],
            candidate["payload"]["tensor_sha256"],
        )
        total_payload_bytes += payload_bytes
        total_payload_tensors += payload_tensors
        layers.append(
            {
                "layer_index": layer_index,
                "payload_sha256": payload_hash,
                "payload_bytes": payload_bytes,
                "payload_tensor_count": payload_tensors,
                "reference_metadata_sha256": sha256_file(reference_metadata_path),
                "candidate_metadata_sha256": sha256_file(candidate_metadata_path),
                "stage_elapsed_seconds": candidate["stage_elapsed_seconds"],
            }
        )

    if len(candidate_config_hashes) != 1 or len(candidate_implementation_hashes) != 1:
        raise ValueError("candidate config or implementation identity drifted across layers")
    result = {
        "schema_version": 1,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": args.arm,
        "last_layer": args.last_layer,
        "layer_count": len(layers),
        "reference_root": str(args.reference_root),
        "candidate_root": str(args.candidate_root),
        "candidate_config_sha256": next(iter(candidate_config_hashes)),
        "candidate_implementation_sha256": next(iter(candidate_implementation_hashes)),
        "total_payload_bytes": total_payload_bytes,
        "total_payload_tensors": total_payload_tensors,
        "all_payloads_bit_exact": True,
        "all_algorithm_metadata_exact": True,
        "layers": layers,
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_FULL_ARTIFACT_COMPARISON_PASSED={args.arm}:0-{args.last_layer}")
    print(f"FLUXBIN_FULL_ARTIFACT_COMPARISON_RESULT={args.output}")


if __name__ == "__main__":
    main()
