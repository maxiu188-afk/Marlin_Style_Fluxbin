#!/usr/bin/env python3
"""Matched BF16, global two-base, and hybrid-s8 WikiText-2 PPL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fluxbin_style import (
    TWO_BASE_PACKED_FORMAT,
    atomic_json,
    materialize_global_two_base_weight,
    materialize_hybrid_s8_weight,
    qwen3_linear_weight_names,
    sha256_file,
    sha256_token_sequences,
    summed_cross_entropy_fp32,
    tensor_sha256,
    unpack_two_bases,
)


PAYLOAD_TENSORS = {
    "global_sign_codes",
    "global_row_scales",
    "global_column_scales",
    "refinement_indices",
    "refinement_sign_codes",
    "refinement_row_scales",
    "refinement_column_scales",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--reconstruction-result", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--accepted-bf16-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def load_protocol(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[list[int]], dict[str, Any]]:
    accepted = config["accepted_protocol"]
    if sha256_file(args.protocol_manifest) != accepted["protocol_manifest_sha256"]:
        raise ValueError("accepted protocol manifest hash drifted")
    if sha256_file(args.token_artifact) != accepted["token_artifact_sha256"]:
        raise ValueError("accepted token artifact hash drifted")
    manifest = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("accepted protocol manifest is not passed")
    protocol = manifest["owned_protocol"]
    expected_fields = (
        "id",
        "sequence_length",
        "token_count",
        "tokens_sha256",
        "full_block_count",
        "blocks_sha256",
        "scored_transition_count",
        "token_artifact_dtype",
    )
    for name in expected_fields:
        if protocol.get(name) != accepted[name]:
            raise ValueError(f"accepted protocol field drifted: {name}")
    token_record = manifest["token_artifact"]
    if token_record["sha256"] != accepted["token_artifact_sha256"]:
        raise ValueError("token artifact record hash drifted")
    if token_record["tensor_sha256"] != accepted["token_artifact_tensor_sha256"]:
        raise ValueError("token tensor record hash drifted")
    with safe_open(args.token_artifact, framework="pt", device="cpu") as source:
        if set(source.keys()) != {"tokens"}:
            raise ValueError("token artifact inventory drifted")
        token_tensor = source.get_tensor("tokens")
    if str(token_tensor.dtype) != accepted["token_artifact_dtype"]:
        raise ValueError("token artifact dtype drifted")
    if tensor_sha256(token_tensor) != accepted["token_artifact_tensor_sha256"]:
        raise ValueError("token tensor hash drifted")
    tokens = token_tensor.tolist()
    if len(tokens) != accepted["token_count"]:
        raise ValueError("token count drifted")
    if sha256_token_sequences([tokens]) != accepted["tokens_sha256"]:
        raise ValueError("token stream hash drifted")
    sequence_length = accepted["sequence_length"]
    used_token_count = protocol["used_token_count"]
    blocks = [
        tokens[start : start + sequence_length]
        for start in range(0, used_token_count, sequence_length)
    ]
    if len(blocks) != accepted["full_block_count"]:
        raise ValueError("full block count drifted")
    if sha256_token_sequences(blocks) != accepted["blocks_sha256"]:
        raise ValueError("block stream hash drifted")
    for name, expected_hash in accepted["tokenizer_files_sha256"].items():
        path = args.snapshot_root / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"tokenizer asset drifted: {name}")
    return blocks, manifest


def validate_bf16_reference(config: dict[str, Any], path: Path) -> dict[str, Any]:
    accepted = config["accepted_bf16_reference"]
    if sha256_file(path) != accepted["result_sha256"]:
        raise ValueError("accepted BF16 reference hash drifted")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed_pending_review":
        raise ValueError("accepted BF16 result status drifted")
    if result["arms"]["bf16"]["perplexity"] != accepted["perplexity"]:
        raise ValueError("accepted BF16 perplexity drifted")
    protocol = config["accepted_protocol"]
    if result["protocol"]["tokens_sha256"] != protocol["tokens_sha256"]:
        raise ValueError("accepted BF16 token stream drifted")
    if result["protocol"]["blocks_sha256"] != protocol["blocks_sha256"]:
        raise ValueError("accepted BF16 block stream drifted")
    return result


def validate_ledger(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any], Path]]]:
    accepted = config["decomposition"]
    if sha256_file(args.reconstruction_result) != accepted["accepted_result_sha256"]:
        raise ValueError("accepted reconstruction result hash drifted")
    result = json.loads(args.reconstruction_result.read_text(encoding="utf-8"))
    expected_top = {
        "status": "passed",
        "config_sha256": accepted["accepted_config_sha256"],
        "implementation_sha256": accepted["accepted_implementation_sha256"],
        "source_manifest_sha256": accepted["accepted_source_manifest_sha256"],
        "next_stage": "not_launched",
    }
    for name, expected in expected_top.items():
        if result.get(name) != expected:
            raise ValueError(f"accepted reconstruction field drifted: {name}")
    contract = result["contract"]
    if contract["payload_format"] != TWO_BASE_PACKED_FORMAT:
        raise ValueError("payload format drifted")
    if contract["stored_arms"] != accepted["stored_arms"]:
        raise ValueError("stored arm inventory drifted")
    if not contract["global_arm_shared_by_pure_and_hybrid"]:
        raise ValueError("global arm sharing contract drifted")
    aggregate = result["aggregate"]
    if aggregate["tensor_count"] != accepted["accepted_tensor_count"]:
        raise ValueError("accepted tensor count drifted")
    if aggregate["parameter_count"] != accepted["accepted_parameter_count"]:
        raise ValueError("accepted parameter count drifted")
    if aggregate["every_tensor_strictly_improved"] is not accepted[
        "accepted_every_tensor_strictly_improved"
    ]:
        raise ValueError("strict improvement record drifted")
    if result["solver_summary"]["global"]["converged_reasons"].get("max_iters", 0) != accepted[
        "accepted_global_max_iters_tensor_count"
    ]:
        raise ValueError("global max-iteration count drifted")
    if result["solver_summary"]["refinement"]["converged_reasons"].get(
        "max_iters", 0
    ) != accepted["accepted_refinement_max_iters_tensor_count"]:
        raise ValueError("refinement max-iteration count drifted")
    if sha256_file(args.snapshot_root / "config.json") != result["model"]["config_sha256"]:
        raise ValueError("model config hash drifted")
    if sha256_file(args.snapshot_root / "model.safetensors.index.json") != result[
        "model"
    ]["index_sha256"]:
        raise ValueError("model index hash drifted")
    expected_names = qwen3_linear_weight_names(config["model"]["expected_hidden_layers"])
    if tuple(record["tensor_name"] for record in result["tensors"]) != expected_names:
        raise ValueError("reconstruction tensor order or coverage drifted")

    records: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for record in result["tensors"]:
        metadata_path = args.artifact_dir / record["metadata_file"]
        payload_path = args.artifact_dir / record["payload_file"]
        if sha256_file(metadata_path) != record["metadata_sha256"]:
            raise ValueError(f"metadata hash drifted: {record['tensor_name']}")
        if payload_path.stat().st_size != record["payload_bytes"]:
            raise ValueError(f"payload size drifted: {record['tensor_name']}")
        if sha256_file(payload_path) != record["payload_sha256"]:
            raise ValueError(f"payload hash drifted: {record['tensor_name']}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "status": "passed",
            "tensor_index": record["tensor_index"],
            "tensor_name": record["tensor_name"],
            "shape": record["shape"],
            "target_sha256": record["target_sha256"],
            "config_sha256": accepted["accepted_config_sha256"],
            "implementation_sha256": accepted["accepted_implementation_sha256"],
            "model_revision": config["model"]["revision"],
            "next_stage": "not_launched",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"metadata drifted for {record['tensor_name']}: {name}")
        if metadata["payload"]["sha256"] != record["payload_sha256"]:
            raise ValueError(f"metadata payload hash drifted: {record['tensor_name']}")
        if metadata["payload"]["stored_arms"] != accepted["stored_arms"]:
            raise ValueError(f"metadata arm inventory drifted: {record['tensor_name']}")
        records.append((record, metadata, payload_path))
    return result, records


def load_payload_tensors(
    metadata: dict[str, Any],
    payload_path: Path,
    *,
    validate_hashes: bool,
) -> dict[str, torch.Tensor]:
    with safe_open(payload_path, framework="pt", device="cpu") as payload:
        if set(payload.keys()) != PAYLOAD_TENSORS:
            raise ValueError(f"payload inventory drifted: {metadata['tensor_name']}")
        tensors = {name: payload.get_tensor(name) for name in payload.keys()}
    if validate_hashes:
        for name, expected_hash in metadata["payload"]["tensor_sha256"].items():
            if tensor_sha256(tensors[name]) != expected_hash:
                raise ValueError(f"payload tensor hash drifted: {metadata['tensor_name']}:{name}")
        decoded = metadata["payload"]["decoded_sign_sha256"]
        if tensor_sha256(unpack_two_bases(tensors["global_sign_codes"])) != decoded[
            "global_signs"
        ]:
            raise ValueError(f"global sign decode drifted: {metadata['tensor_name']}")
        if tensor_sha256(unpack_two_bases(tensors["refinement_sign_codes"])) != decoded[
            "refinement_signs"
        ]:
            raise ValueError(f"refinement sign decode drifted: {metadata['tensor_name']}")
    return tensors


def apply_arm(
    model: torch.nn.Module,
    records: list[tuple[dict[str, Any], dict[str, Any], Path]],
    *,
    arm: str,
    group_size: int,
    columns_per_group: int,
    device: torch.device,
    validated_payloads: set[Path],
) -> dict[str, Any]:
    if arm not in ("global_two_base_rank_one", "hybrid_s8"):
        raise ValueError(f"unsupported arm: {arm}")
    started = time.monotonic()
    parameter_count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, (record, metadata, payload_path) in enumerate(records):
            validate_hashes = payload_path not in validated_payloads
            tensors = load_payload_tensors(
                metadata,
                payload_path,
                validate_hashes=validate_hashes,
            )
            if validate_hashes:
                validated_payloads.add(payload_path)
            if arm == "global_two_base_rank_one":
                weight = materialize_global_two_base_weight(
                    tensors["global_sign_codes"],
                    tensors["global_row_scales"],
                    tensors["global_column_scales"],
                    group_size=group_size,
                    device=device,
                    output_dtype=torch.bfloat16,
                )
            else:
                weight = materialize_hybrid_s8_weight(
                    tensors["global_sign_codes"],
                    tensors["global_row_scales"],
                    tensors["global_column_scales"],
                    tensors["refinement_indices"],
                    tensors["refinement_sign_codes"],
                    tensors["refinement_row_scales"],
                    tensors["refinement_column_scales"],
                    group_size=group_size,
                    columns_per_group=columns_per_group,
                    device=device,
                    output_dtype=torch.bfloat16,
                )
            module_name = record["tensor_name"].removesuffix(".weight")
            module = model.get_submodule(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"target is not Linear: {module_name}")
            if list(module.weight.shape) != record["shape"]:
                raise ValueError(f"model weight shape drifted: {record['tensor_name']}")
            if module.weight.dtype != torch.bfloat16 or module.weight.device != device:
                raise ValueError(f"model weight placement drifted: {record['tensor_name']}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"materialized weight is non-finite: {record['tensor_name']}")
            module.weight.copy_(weight)
            parameter_count += module.weight.numel()
            del tensors, weight
            if (index + 1) % 28 == 0 or index + 1 == len(records):
                print(
                    f"FLUXBIN_{arm.upper()}_MATERIALIZED={index + 1}/{len(records)}",
                    flush=True,
                )
    torch.cuda.synchronize()
    return {
        "tensor_count": len(records),
        "parameter_count": parameter_count,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validated_payload_count": len(validated_payloads),
        "all_payloads_validated": len(validated_payloads) == len(records),
    }


def finite_perplexity(mean_nll: float | None) -> float | None:
    if mean_nll is None or not math.isfinite(mean_nll):
        return None
    try:
        value = math.exp(mean_nll)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def score_model(
    model: torch.nn.Module,
    blocks: list[list[int]],
    *,
    arm: str,
    device: torch.device,
    logit_chunk_tokens: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    block_nll: list[float] = []
    nonfinite_blocks: list[int] = []
    scored_transitions = 0
    logits_dtype = ""
    with torch.inference_mode():
        for block_index, block in enumerate(blocks):
            input_ids = torch.tensor(block, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1, :]
            logits_dtype = str(logits.dtype)
            labels = input_ids[0, 1:]
            chunk_values: list[float] = []
            for start in range(0, logits.shape[0], logit_chunk_tokens):
                stop = min(start + logit_chunk_tokens, logits.shape[0])
                chunk_nll = summed_cross_entropy_fp32(
                    logits[start:stop],
                    labels[start:stop],
                )
                value = float(chunk_nll)
                if not math.isfinite(value):
                    nonfinite_blocks.append(block_index)
                    break
                chunk_values.append(value)
            if block_index not in nonfinite_blocks:
                block_nll.append(math.fsum(chunk_values))
            scored_transitions += len(block) - 1
            del input_ids, logits, labels
            if (block_index + 1) % 10 == 0 or block_index + 1 == len(blocks):
                print(
                    f"FLUXBIN_{arm.upper()}_PPL_PROGRESS={block_index + 1}/{len(blocks)}",
                    flush=True,
                )
    total_nll = math.fsum(block_nll) if not nonfinite_blocks else None
    mean_nll = total_nll / scored_transitions if total_nll is not None else None
    perplexity = finite_perplexity(mean_nll)
    return {
        "block_count": len(blocks),
        "scored_transition_count": scored_transitions,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "metrics_valid": perplexity is not None and not nonfinite_blocks,
        "nonfinite_block_indices": nonfinite_blocks,
        "elapsed_seconds": time.monotonic() - started,
        "logits_dtype": logits_dtype,
        "nll_logits_dtype": "torch.float32",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported PPL config schema")
    if config["evaluation"]["arms"] != [
        "bf16",
        "global_two_base_rank_one",
        "hybrid_s8",
    ]:
        raise ValueError("evaluation arm contract drifted")
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
    blocks, protocol_manifest = load_protocol(config, args)
    bf16_reference = validate_bf16_reference(config, args.accepted_bf16_result)
    ledger_result, records = validate_ledger(config, args)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
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
    validated_payloads: set[Path] = set()
    for arm in ("global_two_base_rank_one", "hybrid_s8"):
        materialization[arm] = apply_arm(
            model,
            records,
            arm=arm,
            group_size=int(config["decomposition"]["group_size"]),
            columns_per_group=int(config["decomposition"]["columns_per_group"]),
            device=device,
            validated_payloads=validated_payloads,
        )
        arms[arm] = score_model(
            model,
            blocks,
            arm=arm,
            device=device,
            logit_chunk_tokens=logit_chunk_tokens,
        )

    expected_transitions = config["accepted_protocol"]["scored_transition_count"]
    counts_match = all(
        arm["scored_transition_count"] == expected_transitions for arm in arms.values()
    )
    metrics_valid = counts_match and all(arm["metrics_valid"] for arm in arms.values())
    bf16_ppl = arms["bf16"]["perplexity"]
    reference = config["accepted_bf16_reference"]
    bf16_difference = (
        abs(bf16_ppl - reference["perplexity"]) if bf16_ppl is not None else None
    )
    bf16_reproduced = (
        bf16_difference is not None and bf16_difference <= reference["absolute_tolerance"]
    )
    payloads_validated = all(
        value["all_payloads_validated"] for value in materialization.values()
    )
    execution_valid = metrics_valid and bf16_reproduced and payloads_validated
    global_ppl = arms["global_two_base_rank_one"]["perplexity"]
    hybrid_ppl = arms["hybrid_s8"]["perplexity"]
    result = {
        "schema_version": 1,
        "status": "completed_pending_review" if execution_valid else "completed_invalid_metrics",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "source_manifest": {
            "path": str(args.source_manifest),
            "sha256": sha256_file(args.source_manifest),
        },
        "protocol_manifest": {
            "path": str(args.protocol_manifest),
            "sha256": sha256_file(args.protocol_manifest),
        },
        "token_artifact": {
            "path": str(args.token_artifact),
            "sha256": sha256_file(args.token_artifact),
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
        "reconstruction_ledger": {
            "path": str(args.reconstruction_result),
            "sha256": config["decomposition"]["accepted_result_sha256"],
            "artifact_dir": str(args.artifact_dir),
            "tensor_count": len(records),
            "parameter_count": ledger_result["aggregate"]["parameter_count"],
            "payload_format": ledger_result["contract"]["payload_format"],
            "stored_arms": ledger_result["contract"]["stored_arms"],
            "all_file_hashes_validated": True,
            "all_internal_payload_hashes_validated": payloads_validated,
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
            "global_minus_bf16_perplexity": (
                global_ppl - bf16_ppl
                if global_ppl is not None and bf16_ppl is not None
                else None
            ),
            "hybrid_minus_bf16_perplexity": (
                hybrid_ppl - bf16_ppl
                if hybrid_ppl is not None and bf16_ppl is not None
                else None
            ),
            "hybrid_minus_global_perplexity": (
                hybrid_ppl - global_ppl
                if hybrid_ppl is not None and global_ppl is not None
                else None
            ),
            "hybrid_relative_perplexity_change_vs_global": (
                (hybrid_ppl - global_ppl) / global_ppl
                if hybrid_ppl is not None and global_ppl is not None
                else None
            ),
            "bf16_reference_absolute_difference": bf16_difference,
        },
        "acceptance_checks": {
            "identical_accepted_token_artifact": True,
            "identical_scored_transition_count": counts_match,
            "all_metrics_finite": metrics_valid,
            "accepted_bf16_reproduced_within_tolerance": bf16_reproduced,
            "all_reconstruction_files_validated": True,
            "all_internal_payload_hashes_validated": payloads_validated,
            "quantized_improvement_required_for_valid_execution": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_PPL_RESULT={args.output}")
    print(f"FLUXBIN_PPL_STATUS={result['status']}")
    print(f"FLUXBIN_PPL_BF16={bf16_ppl}")
    print(f"FLUXBIN_PPL_GLOBAL={global_ppl}")
    print(f"FLUXBIN_PPL_HYBRID_S8={hybrid_ppl}")
    print("FLUXBIN_PPL_READY_FOR_REVIEW")


if __name__ == "__main__":
    main()
