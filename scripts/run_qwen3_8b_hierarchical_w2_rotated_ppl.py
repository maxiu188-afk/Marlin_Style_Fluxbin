#!/usr/bin/env python3
"""Frozen WikiText2 PPL for the offline-rotated hierarchical-W2 H2.50 arm.

Two arms are scored in the same job on the same device: the decoded W3 g128
reference and the rotated H2.50 artifact.  Each arm gets a freshly loaded BF16
model, because the rotation is an in-place reparameterization of the embedding,
`lm_head` and both RMSNorms, not something that can be layered on top of an
already-quantized model.

The unrotated H2.50 value is not re-scored here.  It is carried in as a frozen
baseline from the accepted endpoint run and is only valid for comparison
because this job replays W3 on the same device with the same protocol.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from fluxbin_style import QWEN3_LINEAR_MODULES, atomic_json, sha256_file
from fluxbin_style.rate_distortion import EXPECTED_LAYERS, EXPECTED_LINEARS, EXPECTED_WEIGHTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_8b_w3_rate_distortion_ppl import apply_gptq_dense  # noqa: E402
from run_qwen3_two_base_rank1_s8_ppl import load_protocol, score_model  # noqa: E402
from run_qwen3_8b_hierarchical_w2_gptq import (  # noqa: E402
    CROSS_DEVICE_EXECUTION_POLICY,
    EXECUTION_POLICIES,
    FORMAL_EXECUTION_POLICY,
)
from run_qwen3_8b_hierarchical_w2_ppl import (  # noqa: E402
    apply_hierarchical,
    validate_runtime,
    validate_snapshot,
    validate_w3,
)
from run_qwen3_8b_hierarchical_w2_rotated_gptq import (  # noqa: E402
    ARM,
    runtime_identity,
    rotate_model,
    source_hash,
    validate_config,
)
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
W3_ARM = "gptq_w3_g128_sym"
ARMS = (W3_ARM, ARM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--w3-manifest", type=Path, required=True)
    parser.add_argument("--w3-dir", type=Path, required=True)
    parser.add_argument("--rotated-result", type=Path, required=True)
    parser.add_argument("--rotated-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-policy", choices=EXECUTION_POLICIES, default=FORMAL_EXECUTION_POLICY
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_rotated_result(
    config: dict[str, Any],
    *,
    result_path: Path,
    artifact_dir: Path,
    config_hash: str,
    implementation_hash: str,
    execution_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads(result_path.read_text())
    expected = {
        "status": "completed_pending_ppl",
        "arm": ARM,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"rotated result drifted: {key}")
    if result.get("variant") != config["variants"][ARM]:
        raise ValueError("variant drifted")
    if result.get("rotation", {}).get("format") != config["rotation"]["format"]:
        raise ValueError("rotation format drifted")
    if result.get("execution", {}).get("execution_policy") != execution_policy:
        raise ValueError("execution policy drifted")
    expected_runtime_identity = runtime_identity(result["execution"])
    coverage = result.get("coverage", {})
    if (
        coverage.get("layer_count") != EXPECTED_LAYERS
        or coverage.get("linear_count") != EXPECTED_LINEARS
        or coverage.get("quantized_weight_count") != EXPECTED_WEIGHTS
        or coverage.get("lm_head_quantized") is not False
        or coverage.get("no_fallback") is not True
    ):
        raise ValueError("coverage drifted")
    records = []
    for layer in range(EXPECTED_LAYERS):
        directory = artifact_dir / f"layer-{layer:03d}"
        metadata_path = directory / "metadata.json"
        payload_path = directory / "payload.safetensors"
        source = result["artifacts"][layer]
        if sha256_file(metadata_path) != source["metadata_sha256"]:
            raise ValueError(f"metadata drifted: {layer}")
        if sha256_file(payload_path) != source["payload_sha256"]:
            raise ValueError(f"payload drifted: {layer}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "passed" or metadata.get("arm") != ARM:
            raise ValueError(f"layer metadata invalid: {layer}")
        if metadata.get("rotation_format") != config["rotation"]["format"]:
            raise ValueError(f"layer rotation format drifted: {layer}")
        if metadata.get("runtime_identity") != expected_runtime_identity:
            raise ValueError(f"layer runtime identity drifted: {layer}")
        if [item.get("module") for item in metadata.get("linears", [])] != list(QWEN3_LINEAR_MODULES):
            raise ValueError(f"Linear inventory drifted: {layer}")
        records.append({"layer": layer, "metadata": metadata, "payload": payload_path})
    return result, records


def load_bf16_model(config: dict[str, Any], snapshot_root: Path, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["evaluation"]["attention_implementation"],
    ).to(device).eval()
    model.config.use_cache = False
    return model


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config)
    validate_snapshot(config, args.snapshot_root)
    protocol_config = {"accepted_protocol": config["evaluation"]["accepted_protocol"]}
    blocks, protocol_manifest = load_protocol(protocol_config, args)
    w3_manifest, w3_records = validate_w3(config, args.w3_manifest, args.w3_dir)
    config_hash = sha256_file(args.config)
    implementation_hash, _ = source_hash()
    rotated_result, rotated_records = validate_rotated_result(
        config,
        result_path=args.rotated_result,
        artifact_dir=args.rotated_artifact_dir,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
        execution_policy=args.execution_policy,
    )
    if args.validate_only:
        print("ROTATED_W2_PPL_PREFLIGHT=passed; no PPL launched", flush=True)
        return

    device, runtime = validate_runtime(config, execution_policy=args.execution_policy)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    metrics: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    applied_rotation: dict[str, Any] = {}
    for arm in ARMS:
        model = load_bf16_model(config, args.snapshot_root, device)
        if arm == W3_ARM:
            coverage[arm] = apply_gptq_dense(model, w3_records, device=device)
        else:
            # The payloads live in the rotated basis, so the embedding, both
            # RMSNorms and lm_head must be rotated before the Linears are
            # overwritten; only then is the decoded model self-consistent.
            applied_rotation = rotate_model(model, config, device=device)
            coverage[arm] = apply_hierarchical(model, rotated_records, device=device)
        metrics[arm] = score_model(
            model,
            blocks,
            arm=arm,
            device=device,
            logit_chunk_tokens=config["evaluation"]["logit_chunk_tokens"],
        )
        del model
        torch.cuda.empty_cache()
        atomic_json(
            args.output.with_suffix(".progress.json"),
            {"status": "in_progress", "metrics": metrics, "coverage": coverage},
        )
        print(f"ROTATED_W2_PPL={arm}:{metrics[arm]['perplexity']}", flush=True)

    reference_ppl = metrics[W3_ARM]["perplexity"]
    rotated_ppl = metrics[ARM]["perplexity"]
    baselines = config["baselines"]
    unrotated_ppl = baselines["unrotated_h2_50_perplexity"]
    valid = all(
        metrics[arm].get("metrics_valid")
        and metrics[arm].get("scored_transition_count")
        == config["evaluation"]["accepted_protocol"]["scored_transition_count"]
        and math.isfinite(metrics[arm]["perplexity"])
        for arm in ARMS
    )
    w3_replay_delta = reference_ppl - baselines["w3_endpoint_same_run_perplexity"]
    result = {
        "schema_version": 1,
        "status": (
            "completed_pending_review"
            if valid and args.execution_policy == FORMAL_EXECUTION_POLICY
            else "completed_cross_device_pending_review"
            if valid and args.execution_policy == CROSS_DEVICE_EXECUTION_POLICY
            else "completed_invalid_metrics"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "quantization_implementation_sha256": implementation_hash,
        "evaluation_source_files_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "src/fluxbin_style/hierarchical_w2.py",
                "src/fluxbin_style/offline_rotation.py",
                "scripts/run_qwen3_8b_hierarchical_w2_rotated_gptq.py",
                "scripts/run_qwen3_8b_hierarchical_w2_rotated_ppl.py",
                "scripts/run_qwen3_8b_hierarchical_w2_ppl.py",
                "scripts/run_qwen3_8b_w3_rate_distortion_ppl.py",
                "scripts/run_qwen3_two_base_rank1_s8_ppl.py",
            )
        },
        "protocol": config["evaluation"]["accepted_protocol"],
        "protocol_manifest": protocol_manifest,
        "runtime": runtime,
        "rotation": applied_rotation,
        "w3_reference": {
            **config["w3_reference"],
            "same_run_perplexity": reference_ppl,
            "manifest_sha256": sha256_file(args.w3_manifest),
            "raw_quantized_tensor_bytes": w3_manifest["raw_artifact"]["quantized_tensor_bytes"],
            "delta_versus_endpoint_replay": w3_replay_delta,
        },
        "metrics": metrics,
        "coverage": coverage,
        "comparison": {
            "nominal_bpw": rotated_result["storage"]["nominal_bpw"],
            "actual_bpw_including_permutation_and_steps": rotated_result["storage"][
                "actual_bpw_including_permutation_and_steps"
            ],
            "rotated_perplexity": rotated_ppl,
            "unrotated_perplexity": unrotated_ppl,
            "absolute_gain_versus_unrotated": unrotated_ppl - rotated_ppl,
            "relative_gain_versus_unrotated": 1.0 - rotated_ppl / unrotated_ppl,
            "absolute_ppl_gap_vs_w3": rotated_ppl - reference_ppl,
            "relative_ppl_gap_vs_w3": rotated_ppl / reference_ppl - 1.0,
            "bf16_reference_perplexity": baselines["bf16_reference_perplexity"],
            "recovered_fraction_of_unrotated_to_w3_gap": (
                (unrotated_ppl - rotated_ppl) / (unrotated_ppl - reference_ppl)
                if unrotated_ppl > reference_ppl
                else None
            ),
            "quantization_time_seconds": rotated_result["quantization_time_seconds"],
            "mean_weight_reconstruction_mse": rotated_result["reconstruction"][
                "mean_squared_error"
            ],
        },
        "report_only_flags": {
            "below_continuation_threshold": rotated_ppl < 20.0,
            "w3_replay_matches_endpoint_within_0_01": abs(w3_replay_delta) < 0.01,
            "note": (
                "Report-only. The continuation decision is manual; see the rotated "
                "runbook. A cross-device run is never an A100 reproduction."
            ),
        },
        "baselines": baselines,
        "kernel_auto_launch": False,
        "online_hadamard_auto_launch": False,
        "distillation_auto_launch": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(args.output, result)
    print(f"ROTATED_W2_PPL_RESULT={args.output}", flush=True)


if __name__ == "__main__":
    main()
