#!/usr/bin/env python3
"""Matched WikiText-2 PPL for the independent full-Hessian OBQ arms."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fluxbin_style import (
    QWEN3_LINEAR_MODULES,
    TWO_BASE_PACKED_FORMAT,
    atomic_json,
    materialize_global_two_base_weight,
    materialize_hybrid_s8_weight,
    qwen3_linear_weight_names,
    sha256_file,
    tensor_sha256,
)
from run_qwen3_two_base_rank1_s8_ppl import (
    load_protocol,
    score_model,
    validate_bf16_reference,
    validate_runtime,
)


GLOBAL_SUFFIXES = (
    "global_sign_codes",
    "global_row_scales",
    "global_column_scales",
)
REFINEMENT_SUFFIXES = (
    "refinement_indices",
    "refinement_sign_codes",
    "refinement_row_scales",
    "refinement_column_scales",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--pure-result", type=Path, required=True)
    parser.add_argument("--pure-source-manifest", type=Path, required=True)
    parser.add_argument("--pure-artifact-dir", type=Path, required=True)
    parser.add_argument("--hybrid-result", type=Path, required=True)
    parser.add_argument("--hybrid-source-manifest", type=Path, required=True)
    parser.add_argument("--hybrid-artifact-dir", type=Path, required=True)
    parser.add_argument("--accepted-bf16-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expected_payload_keys(arm: str) -> set[str]:
    if arm not in ("pure", "hybrid_s8"):
        raise ValueError(f"unsupported arm: {arm}")
    suffixes = GLOBAL_SUFFIXES + (REFINEMENT_SUFFIXES if arm == "hybrid_s8" else ())
    return {
        f"{module_name}.{suffix}"
        for module_name in QWEN3_LINEAR_MODULES
        for suffix in suffixes
    }


def validate_arm_ledger(
    config: dict[str, Any],
    *,
    arm: str,
    result_path: Path,
    source_manifest_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decomposition = config["decomposition"]
    accepted = decomposition["arms"][arm]
    if sha256_file(result_path) != accepted["accepted_result_sha256"]:
        raise ValueError(f"accepted {arm} result hash drifted")
    if sha256_file(source_manifest_path) != accepted["accepted_source_manifest_sha256"]:
        raise ValueError(f"accepted {arm} source manifest hash drifted")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version": 2,
        "status": "completed_pending_review",
        "arm": arm,
        "config_sha256": decomposition["accepted_config_sha256"],
        "implementation_sha256": decomposition["accepted_implementation_sha256"],
        "source_manifest_sha256": accepted["accepted_source_manifest_sha256"],
        "next_stage": "not_launched",
    }
    for name, expected in expected_top.items():
        if result.get(name) != expected:
            raise ValueError(f"accepted {arm} result field drifted: {name}")
    if result["model"]["revision"] != config["model"]["revision"]:
        raise ValueError(f"accepted {arm} model revision drifted")
    if (
        result["calibration"]["manifest_sha256"]
        != decomposition["accepted_calibration_manifest_sha256"]
    ):
        raise ValueError(f"accepted {arm} calibration manifest drifted")
    aggregate = result["aggregate"]
    expected_model = config["model"]
    if aggregate["layer_count"] != expected_model["expected_hidden_layers"]:
        raise ValueError(f"accepted {arm} layer count drifted")
    if aggregate["tensor_count"] != expected_model["expected_tensor_count"]:
        raise ValueError(f"accepted {arm} tensor count drifted")
    if aggregate["parameter_count"] != expected_model["expected_parameter_count"]:
        raise ValueError(f"accepted {arm} parameter count drifted")
    if not result["decision"]["execution_valid"]:
        raise ValueError(f"accepted {arm} execution was not valid")
    artifacts = result["artifacts"]
    if len(artifacts) != accepted["expected_layer_count"]:
        raise ValueError(f"accepted {arm} artifact count drifted")

    expected_keys = expected_payload_keys(arm)
    records: list[dict[str, Any]] = []
    tensor_names: list[str] = []
    for layer_index, artifact in enumerate(artifacts):
        if artifact["layer_index"] != layer_index:
            raise ValueError(f"accepted {arm} layer order drifted")
        layer_dir = artifact_dir / f"layer-{layer_index:03d}"
        metadata_path = layer_dir / "metadata.json"
        payload_path = layer_dir / "payload.safetensors"
        if Path(artifact["metadata_path"]) != metadata_path:
            raise ValueError(f"accepted {arm} metadata path drifted")
        if Path(artifact["payload_path"]) != payload_path:
            raise ValueError(f"accepted {arm} payload path drifted")
        if sha256_file(metadata_path) != artifact["metadata_sha256"]:
            raise ValueError(f"accepted {arm} metadata hash drifted: {layer_index}")
        if payload_path.stat().st_size != artifact["payload_bytes"]:
            raise ValueError(f"accepted {arm} payload size drifted: {layer_index}")
        if sha256_file(payload_path) != artifact["payload_sha256"]:
            raise ValueError(f"accepted {arm} payload hash drifted: {layer_index}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "schema_version": 2,
            "status": "passed",
            "arm": arm,
            "layer_index": layer_index,
            "config_sha256": decomposition["accepted_config_sha256"],
            "implementation_sha256": decomposition["accepted_implementation_sha256"],
            "model_revision": config["model"]["revision"],
            "calibration_manifest_sha256": decomposition[
                "accepted_calibration_manifest_sha256"
            ],
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"accepted {arm} metadata drifted: {layer_index}:{name}"
                )
        if metadata["payload"]["format"] != TWO_BASE_PACKED_FORMAT:
            raise ValueError(f"accepted {arm} payload format drifted")
        if not metadata["payload"]["round_trip_verified"]:
            raise ValueError(f"accepted {arm} payload round trip is not verified")
        if metadata["payload"]["sha256"] != artifact["payload_sha256"]:
            raise ValueError(f"accepted {arm} metadata payload hash drifted")
        if set(metadata["payload"]["tensor_sha256"]) != expected_keys:
            raise ValueError(f"accepted {arm} payload inventory drifted: {layer_index}")
        if len(expected_keys) != accepted["expected_payload_tensor_count_per_layer"]:
            raise ValueError(f"configured {arm} payload tensor count drifted")
        if [linear["module"] for linear in metadata["linears"]] != list(
            QWEN3_LINEAR_MODULES
        ):
            raise ValueError(f"accepted {arm} Linear order drifted: {layer_index}")
        for linear in metadata["linears"]:
            tensor_names.append(
                f"model.layers.{layer_index}.{linear['module']}.weight"
            )
        records.append(
            {
                "layer_index": layer_index,
                "metadata": metadata,
                "payload_path": payload_path,
            }
        )
    if tuple(tensor_names) != qwen3_linear_weight_names(
        config["model"]["expected_hidden_layers"]
    ):
        raise ValueError(f"accepted {arm} Linear coverage drifted")
    return result, records


def validate_matched_targets(
    pure_records: list[dict[str, Any]],
    hybrid_records: list[dict[str, Any]],
) -> None:
    if len(pure_records) != len(hybrid_records):
        raise ValueError("arm layer counts differ")
    for pure_layer, hybrid_layer in zip(pure_records, hybrid_records, strict=True):
        if pure_layer["layer_index"] != hybrid_layer["layer_index"]:
            raise ValueError("arm layer order differs")
        for pure, hybrid in zip(
            pure_layer["metadata"]["linears"],
            hybrid_layer["metadata"]["linears"],
            strict=True,
        ):
            for name in ("module", "shape", "parameter_count", "target_bf16_sha256"):
                if pure[name] != hybrid[name]:
                    raise ValueError(
                        f"arm target drifted: {pure_layer['layer_index']}:{pure['module']}:{name}"
                    )


def apply_arm(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    arm: str,
    group_size: int,
    columns_per_group: int,
    device: torch.device,
) -> dict[str, Any]:
    expected_keys = expected_payload_keys(arm)
    started = time.monotonic()
    parameter_count = 0
    tensor_count = 0
    internal_tensor_count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for record in records:
            metadata = record["metadata"]
            payload_path = record["payload_path"]
            with safe_open(payload_path, framework="pt", device="cpu") as source:
                if set(source.keys()) != expected_keys:
                    raise ValueError(
                        f"payload inventory drifted during materialization: {payload_path}"
                    )
                tensors = {name: source.get_tensor(name) for name in source.keys()}
            for name, expected_hash in metadata["payload"]["tensor_sha256"].items():
                if tensor_sha256(tensors[name]) != expected_hash:
                    raise ValueError(
                        f"payload tensor hash drifted: {record['layer_index']}:{name}"
                    )
            internal_tensor_count += len(tensors)
            for linear in metadata["linears"]:
                module_name = linear["module"]
                prefix = f"{module_name}."
                if arm == "pure":
                    weight = materialize_global_two_base_weight(
                        tensors[prefix + "global_sign_codes"],
                        tensors[prefix + "global_row_scales"],
                        tensors[prefix + "global_column_scales"],
                        group_size=group_size,
                        device=device,
                        output_dtype=torch.bfloat16,
                    )
                else:
                    weight = materialize_hybrid_s8_weight(
                        tensors[prefix + "global_sign_codes"],
                        tensors[prefix + "global_row_scales"],
                        tensors[prefix + "global_column_scales"],
                        tensors[prefix + "refinement_indices"],
                        tensors[prefix + "refinement_sign_codes"],
                        tensors[prefix + "refinement_row_scales"],
                        tensors[prefix + "refinement_column_scales"],
                        group_size=group_size,
                        columns_per_group=columns_per_group,
                        device=device,
                        output_dtype=torch.bfloat16,
                    )
                full_module_name = (
                    f"model.layers.{record['layer_index']}.{module_name}"
                )
                module = model.get_submodule(full_module_name)
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(f"target is not Linear: {full_module_name}")
                if list(module.weight.shape) != linear["shape"]:
                    raise ValueError(f"model weight shape drifted: {full_module_name}")
                if module.weight.dtype != torch.bfloat16 or module.weight.device != device:
                    raise ValueError(f"model weight placement drifted: {full_module_name}")
                if not torch.isfinite(weight).all():
                    raise ValueError(f"materialized weight is non-finite: {full_module_name}")
                module.weight.copy_(weight)
                parameter_count += module.weight.numel()
                tensor_count += 1
                del weight
            del tensors
            if (record["layer_index"] + 1) % 8 == 0:
                print(
                    f"FLUXBIN_{arm.upper()}_MATERIALIZED="
                    f"{record['layer_index'] + 1}/{len(records)}",
                    flush=True,
                )
    torch.cuda.synchronize()
    return {
        "layer_count": len(records),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "payload_count": len(records),
        "internal_tensor_count": internal_tensor_count,
        "all_payload_and_internal_tensor_hashes_validated": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("unsupported PPL config schema")
    if config["evaluation"]["arms"] != ["bf16", "pure", "hybrid_s8"]:
        raise ValueError("evaluation arm contract drifted")
    if config["decomposition"]["arms_share_global_payload"]:
        raise ValueError("independent OBQ arms must not share global payloads")
    if config["acceptance"]["backend_auto_launch"]:
        raise ValueError("backend must not auto-launch from PPL")
    if config["acceptance"]["distillation_auto_launch"]:
        raise ValueError("distillation must not auto-launch from PPL")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    if args.snapshot_root.name != config["model"]["revision"]:
        raise ValueError("snapshot path does not match pinned revision")

    device = validate_runtime(config)
    blocks, _ = load_protocol(config, args)
    bf16_reference = validate_bf16_reference(config, args.accepted_bf16_result)
    pure_result, pure_records = validate_arm_ledger(
        config,
        arm="pure",
        result_path=args.pure_result,
        source_manifest_path=args.pure_source_manifest,
        artifact_dir=args.pure_artifact_dir,
    )
    hybrid_result, hybrid_records = validate_arm_ledger(
        config,
        arm="hybrid_s8",
        result_path=args.hybrid_result,
        source_manifest_path=args.hybrid_source_manifest,
        artifact_dir=args.hybrid_artifact_dir,
    )
    validate_matched_targets(pure_records, hybrid_records)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    evaluation = config["evaluation"]
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=evaluation["attention_implementation"],
    )
    if model.config.model_type != config["model"]["model_type"]:
        raise ValueError("loaded model type drifted")
    model.config.use_cache = False
    model.to(device)
    model.eval()
    logit_chunk_tokens = int(evaluation["logit_chunk_tokens"])

    arms: dict[str, dict[str, Any]] = {}
    materialization: dict[str, dict[str, Any]] = {}
    arms["bf16"] = score_model(
        model,
        blocks,
        arm="bf16",
        device=device,
        logit_chunk_tokens=logit_chunk_tokens,
    )
    materialization["pure"] = apply_arm(
        model,
        pure_records,
        arm="pure",
        group_size=int(config["decomposition"]["group_size"]),
        columns_per_group=int(config["decomposition"]["columns_per_group"]),
        device=device,
    )
    arms["pure"] = score_model(
        model,
        blocks,
        arm="pure",
        device=device,
        logit_chunk_tokens=logit_chunk_tokens,
    )
    materialization["hybrid_s8"] = apply_arm(
        model,
        hybrid_records,
        arm="hybrid_s8",
        group_size=int(config["decomposition"]["group_size"]),
        columns_per_group=int(config["decomposition"]["columns_per_group"]),
        device=device,
    )
    arms["hybrid_s8"] = score_model(
        model,
        blocks,
        arm="hybrid_s8",
        device=device,
        logit_chunk_tokens=logit_chunk_tokens,
    )

    expected_transitions = config["accepted_protocol"]["scored_transition_count"]
    counts_match = all(
        arm["scored_transition_count"] == expected_transitions
        for arm in arms.values()
    )
    metrics_valid = counts_match and all(arm["metrics_valid"] for arm in arms.values())
    bf16_ppl = arms["bf16"]["perplexity"]
    pure_ppl = arms["pure"]["perplexity"]
    hybrid_ppl = arms["hybrid_s8"]["perplexity"]
    reference = config["accepted_bf16_reference"]
    bf16_difference = (
        abs(bf16_ppl - reference["perplexity"]) if bf16_ppl is not None else None
    )
    bf16_reproduced = (
        bf16_difference is not None
        and bf16_difference <= reference["absolute_tolerance"]
    )
    payloads_validated = all(
        item["all_payload_and_internal_tensor_hashes_validated"]
        for item in materialization.values()
    )
    expected_tensors = config["model"]["expected_tensor_count"]
    expected_parameters = config["model"]["expected_parameter_count"]
    coverage_valid = all(
        item["layer_count"] == config["model"]["expected_hidden_layers"]
        and item["tensor_count"] == expected_tensors
        and item["parameter_count"] == expected_parameters
        for item in materialization.values()
    )
    execution_valid = (
        metrics_valid
        and bf16_reproduced
        and payloads_validated
        and coverage_valid
    )
    result = {
        "schema_version": 2,
        "status": (
            "completed_pending_review" if execution_valid else "completed_invalid_metrics"
        ),
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "source_manifest": {
            "path": str(args.source_manifest),
            "sha256": sha256_file(args.source_manifest),
        },
        "protocol": {
            "id": config["accepted_protocol"]["id"],
            "token_count": config["accepted_protocol"]["token_count"],
            "tokens_sha256": config["accepted_protocol"]["tokens_sha256"],
            "full_block_count": config["accepted_protocol"]["full_block_count"],
            "blocks_sha256": config["accepted_protocol"]["blocks_sha256"],
            "scored_transition_count": expected_transitions,
            "exact_accepted_artifact_reused": True,
        },
        "decomposition_ledgers": {
            "pure": {
                "path": str(args.pure_result),
                "sha256": config["decomposition"]["arms"]["pure"][
                    "accepted_result_sha256"
                ],
                "artifact_dir": str(args.pure_artifact_dir),
                "aggregate": pure_result["aggregate"],
            },
            "hybrid_s8": {
                "path": str(args.hybrid_result),
                "sha256": config["decomposition"]["arms"]["hybrid_s8"][
                    "accepted_result_sha256"
                ],
                "artifact_dir": str(args.hybrid_artifact_dir),
                "aggregate": hybrid_result["aggregate"],
            },
        },
        "accepted_bf16_reference": {
            **reference,
            "path": str(args.accepted_bf16_result),
            "validated": bf16_reference["status"] == "completed_pending_review",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": __import__("datasets").__version__,
            "safetensors": __import__("safetensors").__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "evaluation": evaluation,
        "arms": arms,
        "materialization": materialization,
        "comparison": {
            "pure_minus_bf16_perplexity": (
                pure_ppl - bf16_ppl
                if pure_ppl is not None and bf16_ppl is not None
                else None
            ),
            "hybrid_minus_bf16_perplexity": (
                hybrid_ppl - bf16_ppl
                if hybrid_ppl is not None and bf16_ppl is not None
                else None
            ),
            "hybrid_minus_pure_perplexity": (
                hybrid_ppl - pure_ppl
                if hybrid_ppl is not None and pure_ppl is not None
                else None
            ),
            "hybrid_relative_perplexity_change_vs_pure": (
                (hybrid_ppl - pure_ppl) / pure_ppl
                if hybrid_ppl is not None and pure_ppl is not None
                else None
            ),
            "bf16_reference_absolute_difference": bf16_difference,
        },
        "acceptance_checks": {
            "identical_accepted_token_artifact": True,
            "identical_scored_transition_count": counts_match,
            "all_metrics_finite": metrics_valid,
            "accepted_bf16_reproduced_within_tolerance": bf16_reproduced,
            "all_layer_payload_and_internal_tensor_hashes_validated": payloads_validated,
            "complete_arm_coverage": coverage_valid,
            "matched_bf16_targets_between_arms": True,
            "quantized_improvement_required_for_valid_execution": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_PPL_RESULT={args.output}")
    print(f"FLUXBIN_PPL_STATUS={result['status']}")
    print(f"FLUXBIN_PPL_BF16={bf16_ppl}")
    print(f"FLUXBIN_PPL_PURE={pure_ppl}")
    print(f"FLUXBIN_PPL_HYBRID_S8={hybrid_ppl}")
    print("FLUXBIN_PPL_READY_FOR_REVIEW")


if __name__ == "__main__":
    main()
