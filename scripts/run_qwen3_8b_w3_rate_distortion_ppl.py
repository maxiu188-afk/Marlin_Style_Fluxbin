#!/usr/bin/env python3
"""Run the four-arm Qwen3-8B W3/QBB study with one frozen BF16 PPL scorer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fluxbin_style import QWEN3_LINEAR_MODULES, atomic_json, sha256_file, tensor_sha256
from fluxbin_style.evaluation import materialize_hybrid_s8_weight
from fluxbin_style.qwen3 import expected_qwen3_linear_shape
from fluxbin_style.qwen3_8b import validate_architecture
from fluxbin_style.rate_distortion import (
    EXPECTED_LAYERS,
    EXPECTED_LINEARS,
    EXPECTED_WEIGHTS,
    FIXED_QBB_SUFFIXES,
    SCALE_SUFFIXES,
    analytical_gptq_storage,
    analytical_qbb_storage,
    build_summary_rows,
    render_summary_markdown,
    summarize_tensor_storage,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_two_base_rank1_s8_ppl import load_protocol, score_model  # noqa: E402
from run_qwen3_8b_w3_gptq import validate_static_config  # noqa: E402
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json"
ARMS = ("bf16", "qbb_current", "gptq_w3_g128_sym", "qbb_fp16_scales")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--qbb-acceptance", type=Path, required=True)
    parser.add_argument("--qbb-result", type=Path, required=True)
    parser.add_argument("--qbb-payload-dir", type=Path, required=True)
    parser.add_argument("--qbb-fp16-result", type=Path, required=True)
    parser.add_argument("--qbb-fp16-dir", type=Path, required=True)
    parser.add_argument("--gptq-manifest", type=Path, required=True)
    parser.add_argument("--gptq-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def expected_qbb_keys() -> set[str]:
    return {
        f"{module}.{suffix}"
        for module in QWEN3_LINEAR_MODULES
        for suffix in (*FIXED_QBB_SUFFIXES, *SCALE_SUFFIXES)
    }


def validate_evaluation_config(config: dict[str, Any]) -> None:
    validate_static_config(config)
    evaluation = config["evaluation"]
    if evaluation["arms"] != list(ARMS):
        raise ValueError("evaluation arm order drifted")
    if evaluation.get("accepted_reproduction_reference") != {
        "result_sha256": "1b1b4353b200a0a0daa357c3e12d3a35544e6fc2f9d16c427e958644baf750ef",
        "bf16_perplexity": 9.724944980689296,
        "qbb_current_perplexity": 13.169788494766266,
        "absolute_tolerance": 0.000001,
    }:
        raise ValueError("accepted BF16/QBB reproduction reference drifted")
    expected_execution = {
        "attention_implementation": "sdpa",
        "use_cache": False,
        "batch_size": 1,
        "logit_chunk_tokens": 128,
        "tf32_allowed": False,
        "dense_decoded_dtype": "torch.bfloat16",
    }
    observed_execution = {name: evaluation.get(name) for name in expected_execution}
    if observed_execution != expected_execution:
        raise ValueError(f"frozen evaluator settings drifted: {observed_execution}")


def validate_runtime(config: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("the formal PPL evaluation requires NVIDIA CUDA")
    torch.cuda.set_device(0)
    expected = config["execution"]
    if torch.cuda.get_device_name(0) not in expected["quality_device_names"]:
        raise RuntimeError("unexpected quality-experiment GPU")
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


def validate_snapshot(config: dict[str, Any], snapshot_root: Path) -> dict[str, Any]:
    model = config["model"]
    if snapshot_root.name != model["revision"]:
        raise ValueError("snapshot directory does not match pinned revision")
    architecture = json.loads((snapshot_root / "config.json").read_text(encoding="utf-8"))
    validate_architecture(architecture)
    for name, digest in model["model_preflight_files"].items():
        if sha256_file(snapshot_root / name) != digest:
            raise ValueError(f"snapshot file drifted: {name}")
    return architecture


def validate_qbb_source(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    qbb = config["qbb"]
    if sha256_file(args.qbb_acceptance) != qbb["accepted_acceptance_sha256"]:
        raise ValueError("current QBB acceptance hash drifted")
    if sha256_file(args.qbb_result) != qbb["accepted_result_sha256"]:
        raise ValueError("current QBB result hash drifted")
    acceptance = json.loads(args.qbb_acceptance.read_text(encoding="utf-8"))
    result = json.loads(args.qbb_result.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted_validation_only" or acceptance.get("result_sha256") != qbb["accepted_result_sha256"]:
        raise ValueError("current QBB source is not accepted")
    payloads = result.get("payloads")
    if not isinstance(payloads, list) or [item.get("layer") for item in payloads] != list(range(EXPECTED_LAYERS)):
        raise ValueError("current QBB layer inventory drifted")
    records = []
    for item in payloads:
        path = args.qbb_payload_dir / f"layer-{item['layer']:03d}.safetensors"
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"current QBB payload hash drifted: {item['layer']}")
        records.append({"layer": item["layer"], "path": path, "sha256": item["sha256"]})
    return records


def validate_qbb_fp16(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads(args.qbb_fp16_result.read_text(encoding="utf-8"))
    expected = {
        "status": "completed_pending_review",
        "arm": "qbb_fp16_scales",
        "source_result_sha256": config["qbb"]["accepted_result_sha256"],
        "config_sha256": sha256_file(args.config),
        "layer_count": EXPECTED_LAYERS,
        "linear_count": EXPECTED_LINEARS,
        "quantized_weight_count": EXPECTED_WEIGHTS,
        "fixed_binary_and_index_hashes_unchanged": True,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"QBB FP16 result drifted: {name}")
    records = []
    if [item.get("layer") for item in result.get("layers", [])] != list(range(EXPECTED_LAYERS)):
        raise ValueError("QBB FP16 layer inventory drifted")
    for item in result["layers"]:
        path = args.qbb_fp16_dir / item["payload"]
        if sha256_file(path) != item["payload_sha256"]:
            raise ValueError(f"QBB FP16 payload hash drifted: {item['layer']}")
        records.append({"layer": item["layer"], "path": path, "sha256": item["payload_sha256"]})
    return result, records


def validate_gptq(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(args.gptq_manifest.read_text(encoding="utf-8"))
    coverage = manifest.get("coverage", {})
    if manifest.get("status") != "completed_pending_review" or manifest.get("config_sha256") != sha256_file(args.config):
        raise ValueError("GPTQ artifact is not bound to this experiment config")
    if coverage != {
        "layer_count": EXPECTED_LAYERS,
        "linear_count": EXPECTED_LINEARS,
        "quantized_weight_count": EXPECTED_WEIGHTS,
        "no_fp_fallback": True,
        "decoded_dtype": "torch.bfloat16",
        "decoded_all_finite": True,
    }:
        raise ValueError("GPTQ coverage or decoded-weight gate failed")
    if manifest["raw_artifact"]["quantized_module_count"] != EXPECTED_LINEARS:
        raise ValueError("GPTQ raw artifact has silent fallback or extra modules")
    records = []
    layers = manifest.get("decoded_layers", [])
    if [item.get("layer") for item in layers] != list(range(EXPECTED_LAYERS)):
        raise ValueError("GPTQ decoded layer inventory drifted")
    for item in layers:
        path = args.gptq_dir / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"GPTQ decoded layer hash drifted: {item['layer']}")
        records.append({**item, "path": path})
    return manifest, records


def inspect_qbb_storage(records: list[dict[str, Any]], *, expected_scale_dtype: torch.dtype) -> dict[str, Any]:
    total_categories: dict[str, int] = {}
    file_bytes = 0
    for record in records:
        path = record["path"]
        file_bytes += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_qbb_keys():
                raise ValueError(f"QBB payload inventory drifted: {record['layer']}")
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        for name, value in tensors.items():
            suffix = name.rsplit(".", 1)[1]
            if suffix in SCALE_SUFFIXES and value.dtype != expected_scale_dtype:
                raise ValueError(f"QBB scale dtype drifted: {record['layer']}:{name}")
        storage = summarize_tensor_storage(tensors)
        if storage["uncategorized"]:
            raise ValueError("QBB storage audit found uncategorized tensors")
        for name, value in storage["categories"].items():
            total_categories[name] = total_categories.get(name, 0) + value
    return {
        "tensor_categories_bytes": dict(sorted(total_categories.items())),
        "serialized_weight_storage_bytes": sum(total_categories.values()),
        "container_file_bytes": file_bytes,
    }


@torch.no_grad()
def apply_qbb(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    tensor_count = 0
    parameter_count = 0
    decoded_hashes: dict[str, str] = {}
    for record in records:
        layer_index = int(record["layer"])
        with safe_open(record["path"], framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        if set(tensors) != expected_qbb_keys():
            raise ValueError(f"QBB payload inventory drifted during apply: {layer_index}")
        for module_name in QWEN3_LINEAR_MODULES:
            prefix = f"{module_name}."
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
                device=device,
                output_dtype=torch.bfloat16,
            )
            full_name = f"model.layers.{layer_index}.{module_name}"
            module = model.get_submodule(full_name)
            if not isinstance(module, torch.nn.Linear) or module.weight.shape != weight.shape:
                raise ValueError(f"QBB target shape/type drifted: {full_name}")
            if weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
                raise ValueError(f"QBB decoded tensor is invalid: {full_name}")
            module.weight.copy_(weight)
            decoded_hashes[full_name] = tensor_sha256(weight)
            tensor_count += 1
            parameter_count += weight.numel()
    return coverage_record(tensor_count, parameter_count, decoded_hashes)


@torch.no_grad()
def apply_gptq_dense(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    tensor_count = 0
    parameter_count = 0
    decoded_hashes: dict[str, str] = {}
    for record in records:
        layer_index = int(record["layer"])
        expected = {f"{module}.weight" for module in QWEN3_LINEAR_MODULES}
        with safe_open(record["path"], framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected:
                raise ValueError(f"GPTQ decoded payload inventory drifted: {layer_index}")
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        tensor_records = {item["module"]: item for item in record["tensors"]}
        for module_name in QWEN3_LINEAR_MODULES:
            weight = tensors[f"{module_name}.weight"]
            full_name = f"model.layers.{layer_index}.{module_name}"
            module = model.get_submodule(full_name)
            if not isinstance(module, torch.nn.Linear) or module.weight.shape != weight.shape:
                raise ValueError(f"GPTQ target shape/type drifted: {full_name}")
            if weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
                raise ValueError(f"GPTQ decoded tensor is invalid: {full_name}")
            digest = tensor_sha256(weight)
            if tensor_records[module_name]["sha256"] != digest:
                raise ValueError(f"GPTQ decoded tensor hash drifted: {full_name}")
            module.weight.copy_(weight.to(device))
            decoded_hashes[full_name] = digest
            tensor_count += 1
            parameter_count += weight.numel()
    return coverage_record(tensor_count, parameter_count, decoded_hashes)


def coverage_record(tensor_count: int, parameter_count: int, hashes: dict[str, str]) -> dict[str, Any]:
    passed = (
        tensor_count == EXPECTED_LINEARS
        and parameter_count == EXPECTED_WEIGHTS
        and len(hashes) == EXPECTED_LINEARS
    )
    if not passed:
        raise RuntimeError("quantized Linear coverage gate failed")
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


def bf16_coverage(model: torch.nn.Module, architecture: dict[str, Any]) -> dict[str, Any]:
    count = 0
    parameters = 0
    for layer_index in range(EXPECTED_LAYERS):
        for module_name in QWEN3_LINEAR_MODULES:
            module = model.get_submodule(f"model.layers.{layer_index}.{module_name}")
            if not isinstance(module, torch.nn.Linear):
                raise TypeError("BF16 target is not Linear")
            if list(module.weight.shape) != expected_qwen3_linear_shape(architecture, module_name):
                raise ValueError("BF16 target shape drifted")
            if module.weight.dtype != torch.bfloat16 or not torch.isfinite(module.weight).all():
                raise ValueError("BF16 target dtype/finite gate failed")
            count += 1
            parameters += module.weight.numel()
    if count != EXPECTED_LINEARS or parameters != EXPECTED_WEIGHTS:
        raise RuntimeError("BF16 Linear coverage gate failed")
    return {
        "status": "passed",
        "layer_count": EXPECTED_LAYERS,
        "linear_count": count,
        "parameter_count": parameters,
        "decoded_dtype": "torch.bfloat16",
        "decoded_all_finite": True,
        "no_silent_fallback": True,
        "decoded_tensor_sha256": {},
    }


def validate_reference_reproduction(
    config: dict[str, Any], arm: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    reference = config["evaluation"]["accepted_reproduction_reference"]
    expected_key = {
        "bf16": "bf16_perplexity",
        "qbb_current": "qbb_current_perplexity",
    }.get(arm)
    if expected_key is None:
        return {"required": False}
    observed = metrics.get("perplexity")
    expected = reference[expected_key]
    difference = abs(observed - expected) if isinstance(observed, (int, float)) else None
    passed = difference is not None and difference <= reference["absolute_tolerance"]
    if not passed:
        raise RuntimeError(
            f"{arm} did not reproduce the accepted same-protocol reference: "
            f"observed={observed}, expected={expected}, difference={difference}"
        )
    return {
        "required": True,
        "passed": True,
        "accepted_result_sha256": reference["result_sha256"],
        "expected_perplexity": expected,
        "observed_perplexity": observed,
        "absolute_difference": difference,
        "absolute_tolerance": reference["absolute_tolerance"],
    }


def write_arm_bundle(
    output_dir: Path,
    *,
    arm: str,
    config: dict[str, Any],
    coverage: dict[str, Any],
    metrics: dict[str, Any],
    storage: dict[str, Any],
    hashes: dict[str, Any],
) -> None:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(arm_dir / "config.json", {"experiment": config, "arm": arm})
    atomic_json(arm_dir / "coverage.json", coverage)
    atomic_json(arm_dir / "eval.json", metrics)
    atomic_json(arm_dir / "storage.json", storage)
    atomic_json(arm_dir / "artifact_hashes.json", hashes)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_evaluation_config(config)
    architecture = validate_snapshot(config, args.snapshot_root)
    protocol_config = {"accepted_protocol": config["evaluation"]["accepted_protocol"]}
    blocks, protocol_manifest = load_protocol(protocol_config, args)
    qbb_records = validate_qbb_source(config, args)
    qbb_fp16_result, qbb_fp16_records = validate_qbb_fp16(config, args)
    gptq_manifest, gptq_records = validate_gptq(config, args)
    if args.validate_only:
        print("FLUXBIN_W3_RATE_DISTORTION_PREFLIGHT=passed; no PPL launched")
        return
    device = validate_runtime(config)
    current_qbb_storage = inspect_qbb_storage(qbb_records, expected_scale_dtype=torch.float32)
    fp16_qbb_storage = inspect_qbb_storage(qbb_fp16_records, expected_scale_dtype=torch.float16)
    qbb32_analytical = analytical_qbb_storage(config["model"], scale_bytes=4, include_lookup=False)
    qbb16_analytical = analytical_qbb_storage(config["model"], scale_bytes=2, include_lookup=False)
    if current_qbb_storage["serialized_weight_storage_bytes"] != qbb32_analytical["persistent_tensor_bytes"]:
        raise RuntimeError("current QBB storage disagrees with analytical accounting")
    if fp16_qbb_storage["serialized_weight_storage_bytes"] != qbb16_analytical["persistent_tensor_bytes"]:
        raise RuntimeError("FP16 QBB storage disagrees with analytical accounting")
    gptq_tensor_bytes = gptq_manifest["raw_artifact"]["quantized_tensor_bytes"]
    storage = {
        "bf16": {
            "nominal_bits": 16,
            "analytical_effective_bits_per_weight": 16.0,
            "effective_bits_per_weight": 16.0,
            "serialized_weight_storage_bytes": EXPECTED_WEIGHTS * 2,
            "container_file_bytes": None,
            "linear_coverage": "252/252",
        },
        "qbb_current": {
            "nominal_bits": "2-base+s8",
            "analytical_effective_bits_per_weight": qbb32_analytical["effective_bits_per_weight"],
            "effective_bits_per_weight": current_qbb_storage["serialized_weight_storage_bytes"] * 8 / EXPECTED_WEIGHTS,
            **current_qbb_storage,
            "linear_coverage": "252/252",
            "legacy_lookup_included": False,
        },
        "gptq_w3_g128_sym": {
            "nominal_bits": 3,
            "analytical_effective_bits_per_weight": analytical_gptq_storage(config["model"])["effective_bits_per_weight"],
            "effective_bits_per_weight": gptq_tensor_bytes * 8 / EXPECTED_WEIGHTS,
            "serialized_weight_storage_bytes": gptq_tensor_bytes,
            "model_artifact_file_bytes": gptq_manifest["raw_artifact"]["serialized_model_artifact_file_bytes"],
            "tensor_categories_bytes": gptq_manifest["raw_artifact"]["tensor_categories_bytes"],
            "tensor_category_dtypes": gptq_manifest["raw_artifact"]["tensor_category_dtypes"],
            "linear_coverage": "252/252",
            "rate_excludes_pass_through_model_tensors": True,
        },
        "qbb_fp16_scales": {
            "nominal_bits": "2-base+s8",
            "analytical_effective_bits_per_weight": qbb16_analytical["effective_bits_per_weight"],
            "effective_bits_per_weight": fp16_qbb_storage["serialized_weight_storage_bytes"] * 8 / EXPECTED_WEIGHTS,
            **fp16_qbb_storage,
            "linear_coverage": "252/252",
            "legacy_lookup_included": False,
        },
    }
    args.output_dir.mkdir(parents=True)
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
    validate_architecture(model.config.to_dict())
    model.config.use_cache = False
    metrics: dict[str, Any] = {}
    coverages: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    artifact_hashes = {
        "bf16": {
            "snapshot_files_sha256": config["model"]["model_preflight_files"],
        },
        "qbb_current": {
            "acceptance_sha256": sha256_file(args.qbb_acceptance),
            "result_sha256": sha256_file(args.qbb_result),
            "payload_sha256": {str(item["layer"]): item["sha256"] for item in qbb_records},
        },
        "gptq_w3_g128_sym": {
            "manifest_sha256": sha256_file(args.gptq_manifest),
            "raw_weight_files": gptq_manifest["raw_artifact"]["weight_files"],
            "decoded_layer_sha256": {str(item["layer"]): item["sha256"] for item in gptq_records},
        },
        "qbb_fp16_scales": {
            "result_sha256": sha256_file(args.qbb_fp16_result),
            "payload_sha256": {str(item["layer"]): item["sha256"] for item in qbb_fp16_records},
        },
    }
    for arm in ARMS:
        if arm == "bf16":
            coverages[arm] = bf16_coverage(model, architecture)
        elif arm == "qbb_current":
            coverages[arm] = apply_qbb(model, qbb_records, device=device)
        elif arm == "gptq_w3_g128_sym":
            coverages[arm] = apply_gptq_dense(model, gptq_records, device=device)
        else:
            coverages[arm] = apply_qbb(model, qbb_fp16_records, device=device)
        metrics[arm] = score_model(
            model,
            blocks,
            arm=arm,
            device=device,
            logit_chunk_tokens=config["evaluation"]["logit_chunk_tokens"],
        )
        reproduction[arm] = validate_reference_reproduction(config, arm, metrics[arm])
        write_arm_bundle(
            args.output_dir,
            arm=arm,
            config=config,
            coverage=coverages[arm],
            metrics=metrics[arm],
            storage=storage[arm],
            hashes=artifact_hashes[arm],
        )
        print(f"FLUXBIN_{arm.upper()}_PPL={metrics[arm]['perplexity']}", flush=True)
    rows = build_summary_rows(metrics, storage)
    summary = {
        "schema_version": 1,
        "status": "completed_pending_effect_size_review",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
        "token_artifact_sha256": sha256_file(args.token_artifact),
        "protocol": protocol_manifest["owned_protocol"],
        "rows": rows,
        "arms": metrics,
        "storage": storage,
        "coverage": coverages,
        "accepted_reference_reproduction": reproduction,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "decision": {
            "status": "manual_review_required",
            "reason": "Use effect size and the user-defined Case A/B/C rules; no small automatic threshold is encoded.",
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        render_summary_markdown(rows, status=summary["status"]),
        encoding="utf-8",
    )
    print(f"FLUXBIN_W3_RATE_DISTORTION_SUMMARY={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
