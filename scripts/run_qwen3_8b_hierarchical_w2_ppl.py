#!/usr/bin/env python3
"""Frozen WikiText2 PPL curve for W3 and four hierarchical-W2 arms."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fluxbin_style import QWEN3_LINEAR_MODULES, atomic_json, sha256_file, tensor_sha256
from fluxbin_style.hierarchical_w2 import materialize_payload, payload_from_tensors
from fluxbin_style.rate_distortion import EXPECTED_LAYERS, EXPECTED_LINEARS, EXPECTED_WEIGHTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_8b_w3_rate_distortion_ppl import apply_gptq_dense  # noqa: E402
from run_qwen3_two_base_rank1_s8_ppl import load_protocol, score_model  # noqa: E402
from run_qwen3_8b_hierarchical_w2_gptq import (  # noqa: E402
    CROSS_DEVICE_EXECUTION_POLICY,
    EXECUTION_POLICIES,
    FORMAL_EXECUTION_POLICY,
    PAYLOAD_FIELDS,
    source_hash,
    validate_config,
    validate_device_policy,
)
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"
HIERARCHICAL_ARMS = ("H2.50", "H2.625", "H2.75", "H2.875")
ARMS = ("gptq_w3_g128_sym", *HIERARCHICAL_ARMS)
ENDPOINT_ARMS = ("H2.50", "H2.875")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--w3-manifest", type=Path, required=True)
    parser.add_argument("--w3-dir", type=Path, required=True)
    for slug in ("h2-50", "h2-625", "h2-75", "h2-875"):
        parser.add_argument(f"--{slug}-result", type=Path)
        parser.add_argument(f"--{slug}-artifact-dir", type=Path)
    parser.add_argument(
        "--evaluation-scope",
        choices=("full-four", "endpoint-only"),
        default="full-four",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-policy",
        choices=EXECUTION_POLICIES,
        default=FORMAL_EXECUTION_POLICY,
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_runtime(
    config: dict[str, Any], *, execution_policy: str
) -> tuple[torch.device, dict[str, Any]]:
    if list(sys.version_info[:2]) != config["execution"]["python_major_minor"]:
        raise RuntimeError(f"Python runtime drifted: {platform.python_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    expected = config["execution"]
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    formal_match = validate_device_policy(
        config,
        device_name=name,
        capability=capability,
        execution_policy=execution_policy,
    )
    observed = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "safetensors": __import__("safetensors").__version__,
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(f"{key} runtime drifted: {value}")
    return torch.device("cuda:0"), {
        "device": name,
        "compute_capability": capability,
        "execution_policy": execution_policy,
        "formal_a100_device_match": formal_match,
        "python": platform.python_version(),
        **observed,
    }


def validate_snapshot(config: dict[str, Any], root: Path) -> None:
    if root.name != config["model"]["revision"]:
        raise ValueError("model revision drifted")
    for name, digest in config["model"]["model_preflight_files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"snapshot content drifted: {name}")


def validate_w3(config: dict[str, Any], manifest_path: Path, decoded_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(manifest_path) != config["w3_reference"]["source_manifest_sha256"]:
        raise ValueError("W3 reference manifest drifted")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed_pending_review":
        raise ValueError("W3 reference is not a complete artifact")
    records = []
    for item in manifest.get("decoded_layers", []):
        layer = int(item["layer"])
        path = decoded_dir / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"W3 decoded layer drifted: {layer}")
        records.append({**item, "path": path})
    if [item["layer"] for item in records] != list(range(EXPECTED_LAYERS)):
        raise ValueError("W3 layer coverage drifted")
    return manifest, records


def selected_hierarchical_arms(evaluation_scope: str) -> tuple[str, ...]:
    if evaluation_scope == "full-four":
        return HIERARCHICAL_ARMS
    if evaluation_scope == "endpoint-only":
        return ENDPOINT_ARMS
    raise ValueError(f"unsupported evaluation scope: {evaluation_scope}")


def hierarchical_arg_pairs(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    all_pairs = {
        "H2.50": (args.h2_50_result, args.h2_50_artifact_dir),
        "H2.625": (args.h2_625_result, args.h2_625_artifact_dir),
        "H2.75": (args.h2_75_result, args.h2_75_artifact_dir),
        "H2.875": (args.h2_875_result, args.h2_875_artifact_dir),
    }
    selected = selected_hierarchical_arms(args.evaluation_scope)
    result: dict[str, tuple[Path, Path]] = {}
    for arm in selected:
        result_path, artifact_dir = all_pairs[arm]
        if result_path is None or artifact_dir is None:
            raise ValueError(f"missing required endpoint input: {arm}")
        result[arm] = (result_path, artifact_dir)
    return result


def validate_hierarchical(
    config: dict[str, Any],
    *,
    arm: str,
    result_path: Path,
    artifact_dir: Path,
    config_hash: str,
    implementation_hash: str,
    execution_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads(result_path.read_text())
    expected = {
        "status": "completed_pending_ppl",
        "arm": arm,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"hierarchical result drifted: {arm}:{key}")
    if result.get("variant") != config["variants"][arm]:
        raise ValueError(f"variant drifted: {arm}")
    if result.get("execution", {}).get("execution_policy") != execution_policy:
        raise ValueError(f"execution policy drifted: {arm}")
    if result.get("coverage") != {
        "layer_count": EXPECTED_LAYERS,
        "linear_count": EXPECTED_LINEARS,
        "quantized_weight_count": EXPECTED_WEIGHTS,
        "lm_head_quantized": False,
        "no_fallback": True,
    }:
        raise ValueError(f"coverage drifted: {arm}")
    records = []
    for layer in range(EXPECTED_LAYERS):
        directory = artifact_dir / f"layer-{layer:03d}"
        metadata_path = directory / "metadata.json"
        payload_path = directory / "payload.safetensors"
        source = result["artifacts"][layer]
        if sha256_file(metadata_path) != source["metadata_sha256"]:
            raise ValueError(f"metadata drifted: {arm}:{layer}")
        if sha256_file(payload_path) != source["payload_sha256"]:
            raise ValueError(f"payload drifted: {arm}:{layer}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "passed" or metadata.get("arm") != arm:
            raise ValueError(f"layer metadata invalid: {arm}:{layer}")
        if [item.get("module") for item in metadata.get("linears", [])] != list(QWEN3_LINEAR_MODULES):
            raise ValueError(f"Linear inventory drifted: {arm}:{layer}")
        records.append({"layer": layer, "metadata": metadata, "payload": payload_path})
    return result, records


@torch.no_grad()
def apply_hierarchical(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    tensor_count = 0
    parameter_count = 0
    hashes: dict[str, str] = {}
    for layer_record in records:
        with safe_open(layer_record["payload"], framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        expected_tensors = {
            f"{module}.{field}" for module in QWEN3_LINEAR_MODULES for field in PAYLOAD_FIELDS
        }
        if set(tensors) != expected_tensors:
            raise ValueError(f"payload tensor inventory drifted: {layer_record['layer']}")
        for record in layer_record["metadata"]["linears"]:
            module_name = record["module"]
            prefix = module_name + "."
            fields = {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}
            payload = payload_from_tensors(
                fields,
                out_features=record["shape"][0],
                in_features=record["shape"][1],
                r32_bits=record["r32_bits"],
                r16_bits=record["r16_bits"],
            )
            weight = materialize_payload(payload, device=device, dtype=torch.bfloat16)
            full_name = f"model.layers.{layer_record['layer']}.{module_name}"
            module = model.get_submodule(full_name)
            if not isinstance(module, torch.nn.Linear) or module.weight.shape != weight.shape:
                raise ValueError(f"target drifted: {full_name}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"decoded weight is non-finite: {full_name}")
            module.weight.copy_(weight)
            hashes[full_name] = tensor_sha256(weight)
            tensor_count += 1
            parameter_count += weight.numel()
    if tensor_count != EXPECTED_LINEARS or parameter_count != EXPECTED_WEIGHTS:
        raise RuntimeError("hierarchical coverage gate failed")
    return {
        "status": "passed",
        "layer_count": EXPECTED_LAYERS,
        "linear_count": tensor_count,
        "parameter_count": parameter_count,
        "decoded_dtype": "torch.bfloat16",
        "decoded_all_finite": True,
        "no_silent_fallback": True,
        "decoded_tensor_sha256": hashes,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config["evaluation"]["arms"] != list(ARMS):
        raise ValueError("evaluation arms drifted")
    selected_arms = selected_hierarchical_arms(args.evaluation_scope)
    arms = ("gptq_w3_g128_sym", *selected_arms)
    validate_snapshot(config, args.snapshot_root)
    protocol_config = {"accepted_protocol": config["evaluation"]["accepted_protocol"]}
    blocks, protocol_manifest = load_protocol(protocol_config, args)
    w3_manifest, w3_records = validate_w3(config, args.w3_manifest, args.w3_dir)
    config_hash = sha256_file(args.config)
    implementation_hash, _ = source_hash()
    hierarchical_results: dict[str, dict[str, Any]] = {}
    hierarchical_records: dict[str, list[dict[str, Any]]] = {}
    for arm, (result_path, artifact_dir) in hierarchical_arg_pairs(args).items():
        hierarchical_results[arm], hierarchical_records[arm] = validate_hierarchical(
            config,
            arm=arm,
            result_path=result_path,
            artifact_dir=artifact_dir,
            config_hash=config_hash,
            implementation_hash=implementation_hash,
            execution_policy=args.execution_policy,
        )
    if args.validate_only:
        print("HIERARCHICAL_W2_PPL_PREFLIGHT=passed; no PPL launched", flush=True)
        return
    device, runtime = validate_runtime(config, execution_policy=args.execution_policy)
    from transformers import AutoModelForCausalLM

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["evaluation"]["attention_implementation"],
    ).to(device).eval()
    model.config.use_cache = False
    metrics: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm == "gptq_w3_g128_sym":
            coverage[arm] = apply_gptq_dense(model, w3_records, device=device)
        else:
            coverage[arm] = apply_hierarchical(model, hierarchical_records[arm], device=device)
        metrics[arm] = score_model(
            model,
            blocks,
            arm=arm,
            device=device,
            logit_chunk_tokens=config["evaluation"]["logit_chunk_tokens"],
        )
        progress = args.output.with_suffix(".progress.json")
        atomic_json(progress, {"status": "in_progress", "metrics": metrics, "coverage": coverage})
        print(f"HIERARCHICAL_W2_PPL={arm}:{metrics[arm]['perplexity']}", flush=True)

    reference_ppl = metrics["gptq_w3_g128_sym"]["perplexity"]
    comparison = {}
    for arm in selected_arms:
        ppl = metrics[arm]["perplexity"]
        result = hierarchical_results[arm]
        comparison[arm] = {
            "nominal_bpw": result["storage"]["nominal_bpw"],
            "actual_bpw_including_permutation_and_steps": result["storage"]["actual_bpw_including_permutation_and_steps"],
            "perplexity": ppl,
            "absolute_ppl_gap_vs_w3": ppl - reference_ppl,
            "relative_ppl_gap_vs_w3": ppl / reference_ppl - 1.0,
            "quantization_time_seconds": result["quantization_time_seconds"],
            "quantization_time_breakdown_seconds": result["time_breakdown_seconds"],
            "mean_weight_reconstruction_mse": result["reconstruction"]["mean_squared_error"],
        }
    ppl_curve = [metrics[arm]["perplexity"] for arm in selected_arms]
    monotonic = all(left >= right for left, right in zip(ppl_curve, ppl_curve[1:]))
    valid = all(
        metrics[arm].get("metrics_valid")
        and metrics[arm].get("scored_transition_count") == config["evaluation"]["accepted_protocol"]["scored_transition_count"]
        and math.isfinite(metrics[arm]["perplexity"])
        for arm in arms
    )
    endpoint_difference = (
        abs(metrics["H2.875"]["perplexity"] - metrics["H2.50"]["perplexity"])
        if args.evaluation_scope == "endpoint-only"
        else None
    )
    result = {
        "schema_version": 1,
        "status": (
            "completed_endpoint_pending_review"
            if valid and args.execution_policy == FORMAL_EXECUTION_POLICY and args.evaluation_scope == "endpoint-only"
            else "completed_cross_device_endpoint_pending_review"
            if valid and args.execution_policy == CROSS_DEVICE_EXECUTION_POLICY and args.evaluation_scope == "endpoint-only"
            else "completed_pending_review"
            if valid and args.execution_policy == FORMAL_EXECUTION_POLICY
            else "completed_cross_device_pending_review"
            if valid and args.execution_policy == CROSS_DEVICE_EXECUTION_POLICY
            else "completed_invalid_metrics"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "quantization_implementation_sha256": implementation_hash,
        "evaluation_source_files_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "src/fluxbin_style/hierarchical_w2.py",
                "scripts/run_qwen3_8b_hierarchical_w2_ppl.py",
                "scripts/run_qwen3_8b_w3_rate_distortion_ppl.py",
                "scripts/run_qwen3_two_base_rank1_s8_ppl.py"
            )
        },
        "protocol": config["evaluation"]["accepted_protocol"],
        "protocol_manifest": protocol_manifest,
        "runtime": runtime,
        "w3_reference": {
            **config["w3_reference"],
            "same_run_perplexity": reference_ppl,
            "manifest_sha256": sha256_file(args.w3_manifest),
            "raw_quantized_tensor_bytes": w3_manifest["raw_artifact"]["quantized_tensor_bytes"],
        },
        "metrics": metrics,
        "coverage": coverage,
        "comparison": comparison,
        "evaluation_scope": args.evaluation_scope,
        "endpoint_test": {
            "absolute_ppl_difference_h2_875_vs_h2_50": endpoint_difference,
            "flat_curve_threshold": 0.05,
            "flat_curve_confirmed": (
                endpoint_difference < 0.05 if endpoint_difference is not None else None
            ),
        },
        "curve": {
            "budget_order": list(selected_arms),
            "perplexity": ppl_curve,
            "monotonically_non_increasing": monotonic,
            "monotonicity_is_report_only_not_an_acceptance_gate": True,
            "intermediate_budgets_intentionally_not_scored": args.evaluation_scope == "endpoint-only",
        },
        "other_accuracy_metrics": [],
        "elapsed_seconds": time.monotonic() - started,
        "decision": {
            "metrics_valid": valid,
            "manual_review_required": True,
            "additional_experiments_auto_launched": False,
            "rotation_auto_launched": False,
            "distillation_auto_launched": False,
            "kernel_auto_launched": False,
        },
    }
    atomic_json(args.output, result)
    print(f"HIERARCHICAL_W2_PPL_RESULT={args.output}", flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
