#!/usr/bin/env python3
"""Run the prepared full-model Qwen3-8B W3 M=1 CUDA Graph trial."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256
from fluxbin_style.full_model_trial import compare_trace, decode_trace, validate_routes
from fluxbin_style.fused_modules import fused_qwen3_modules
from fluxbin_style.gptq_deployment import FIELDS, restore_planar_w3
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.static_decode import StaticDecodeSession, prepared_linears
from fluxbin_style.w3_lut_artifacts import load_w3_lut_layer, replace_w3_model_linears
from fluxbin_style.w3_lut_deployment import (
    QWEN3_ROW_TILE_BY_SHAPE,
    QWEN3_W3_ROUTE_POLICIES,
    load_w3_lut_extension,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/acceleration/qwen3_8b_w3_full_m1_v1.json"
CORRECTED_PROTOCOL = ROOT / "configs/acceleration/qwen3_8b_w3_corrected_full_m1_v1.json"
CALIBRATED_PROTOCOL = ROOT / "configs/acceleration/qwen3_8b_w3_full_m1_v2.json"
# The fused path is not bit-exact with stock, so the calibrated stock/fused
# pair uses distinct v2 protocol ids rather than changing historical v1.
FUSED_CALIBRATED_PROTOCOL = ROOT / "configs/acceleration/qwen3_8b_w3_fused_full_m1_v2.json"

LEGACY_ACCEPTANCE_POLICY = "legacy_absolute_v1"
CALIBRATED_ACCEPTANCE_POLICY = "relative_nrmse_logprob_plus_trace_invariants_v1"


def statistics_row(samples, threshold, *, trim_per_side=0):
    """Timing summary whose stability verdict tolerates isolated host spikes.

    The reported median stays the full-sample median, so speedups are unchanged.
    Only the stability statistic is trimmed: a single interfering sample out of
    ten used to void an entire round under the raw (max-min)/median rule. The
    raw range is still reported so the spike stays visible and auditable.
    """
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("invalid timing samples")
    if trim_per_side < 0:
        raise ValueError("negative trim_per_side")
    median = statistics.median(samples)
    relative_range = (max(samples) - min(samples)) / median
    ordered = sorted(samples)
    # Trimming needs at least three survivors to stay meaningful; below that the
    # raw range is kept so short sample sets are not silently given a free pass.
    applied = trim_per_side if len(ordered) - 2 * trim_per_side >= 3 else 0
    trimmed = ordered[applied : len(ordered) - applied] if applied else ordered
    trimmed_relative_range = (max(trimmed) - min(trimmed)) / median
    return {
        "samples": samples,
        "median": median,
        "relative_range": relative_range,
        "trimmed_relative_range": trimmed_relative_range,
        "trim_per_side": applied,
        "trimmed_sample_count": len(trimmed),
        "stability_rule": "trimmed_relative_range" if applied else "relative_range",
        "stable": trimmed_relative_range <= threshold,
    }


def relative_backend_gate(packed_nrmse, control_nrmse, limit):
    """Gate packed-vs-decoded against the harness's own amplification control.

    The absolute 0.005 logits NRMSE limit is below what this harness produces for
    a zero-quantization control: original_bf16 compared against itself under a
    mathematically equivalent attention-mask reformulation lands at 0.0115-0.0143
    over the complete 32-step logits trace. This gate therefore asks whether the
    packed backend deviates more than the harness deviates from itself, which is
    self-calibrating and needs no fixed absolute threshold. See
    docs/W3_NUMERICAL_GATE_CALIBRATION.md. It does not erase the frozen
    absolute gate: v1 retains it for historical acceptance, while calibrated
    v2 reports it without using it to accept or reject a candidate.
    """
    usable = (
        isinstance(packed_nrmse, (int, float))
        and not isinstance(packed_nrmse, bool)
        and isinstance(control_nrmse, (int, float))
        and not isinstance(control_nrmse, bool)
        and math.isfinite(packed_nrmse)
        and math.isfinite(control_nrmse)
        and control_nrmse > 0
        and isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and math.isfinite(limit)
        and limit > 0
    )
    return {
        "metric": "logits.normalized_rmse",
        "control_source": "original_bf16.dynamic_static_check",
        "control_nrmse": control_nrmse,
        "packed_nrmse": packed_nrmse,
        "ratio": packed_nrmse / control_nrmse if usable else None,
        "limit": limit,
        "passed": bool(usable and packed_nrmse <= control_nrmse * limit),
    }


def relative_logprob_gate(packed_max_error, control_max_error, limit):
    """Calibrate max log-probability error against the same-run BF16 control."""
    usable = (
        isinstance(packed_max_error, (int, float))
        and not isinstance(packed_max_error, bool)
        and math.isfinite(packed_max_error)
        and isinstance(control_max_error, (int, float))
        and not isinstance(control_max_error, bool)
        and math.isfinite(control_max_error)
        and control_max_error > 0
        and isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and math.isfinite(limit)
        and limit > 0
    )
    return {
        "metric": "max_abs_logprob_error",
        "control_source": "original_bf16.dynamic_static_check",
        "control_max_abs_error": control_max_error,
        "packed_max_abs_error": packed_max_error,
        "ratio": packed_max_error / control_max_error if usable else None,
        "limit": limit,
        "passed": bool(usable and packed_max_error <= control_max_error * limit),
    }


def calibrated_backend_gate(
    trace_check,
    control_nrmse,
    nrmse_limit,
    control_logprob_max_error,
    logprob_relative_limit,
):
    """Combine the calibrated NRMSE budget with fail-closed trace invariants.

    The legacy absolute numerical gate remains in ``trace_check['passed']`` for
    auditability, but its fixed 0.005 trace-NRMSE limit is below the measured
    amplification baseline of this harness.  New trials therefore accept the
    backend on the relative gate only when shape/finite metrics, forced context,
    greedy predictions and the same-run relative log-probability bound also
    pass.  This prevents replacing over-strict scalar thresholds with an
    under-specified scalar threshold.
    """
    logits = trace_check.get("logits", {})
    finite_logit_metrics = all(
        isinstance(logits.get(key), (int, float))
        and not isinstance(logits.get(key), bool)
        and math.isfinite(logits[key])
        for key in ("max_abs_error", "reference_rms", "normalized_rmse")
    )
    logprob_error = trace_check.get("max_abs_logprob_error")
    finite_logprob_metric = bool(
        isinstance(logprob_error, (int, float))
        and not isinstance(logprob_error, bool)
        and math.isfinite(logprob_error)
    )
    invariants = {
        "shape_and_finite_logit_metrics": finite_logit_metrics,
        "fed_tokens_equal": trace_check.get("fed_tokens_equal") is True,
        "greedy_tokens_equal": trace_check.get("greedy_tokens_equal") is True,
        "finite_logprob_metric": finite_logprob_metric,
    }
    relative = relative_backend_gate(
        logits.get("normalized_rmse"), control_nrmse, nrmse_limit
    )
    relative_logprob = relative_logprob_gate(
        logprob_error, control_logprob_max_error, logprob_relative_limit
    )
    invariants_passed = all(invariants.values())
    relative_checks_passed = bool(relative["passed"] and relative_logprob["passed"])
    return {
        "policy": "relative_nrmse_logprob_plus_trace_invariants_v1",
        "relative_check": relative,
        "relative_logprob_check": relative_logprob,
        "relative_checks_passed": relative_checks_passed,
        "invariants": invariants,
        "invariants_passed": invariants_passed,
        "legacy_absolute_check_passed": trace_check.get("passed") is True,
        "passed": bool(relative_checks_passed and invariants_passed),
    }


def build_candidate_acceptance(primary_comparisons, route_by_arm, acceptance_policy):
    """Aggregate per-prompt primary checks without vacuous acceptance."""
    candidates = {}
    for packed_arm, route in route_by_arm.items():
        rows = [
            item for item in primary_comparisons if item["packed_arm"] == packed_arm
        ]
        if not rows:
            raise RuntimeError(f"missing primary comparisons for {packed_arm}")
        legacy_passed = all(
            item["packed_vs_decoded_w3_check"]["passed"] for item in rows
        )
        calibrated_passed = all(
            item["packed_vs_decoded_backend_acceptance"]["passed"] for item in rows
        )
        candidate = {
            "route": route,
            "acceptance_policy": acceptance_policy,
            "primary_timings_stable": all(item["stable"] for item in rows),
            "legacy_absolute_backend_checks_passed": legacy_passed,
            "relative_backend_checks_passed": all(
                item["packed_vs_decoded_backend_acceptance"]["relative_checks_passed"]
                for item in rows
            ),
            "backend_invariants_passed": all(
                item["packed_vs_decoded_backend_acceptance"]["invariants_passed"]
                for item in rows
            ),
            "calibrated_backend_checks_passed": calibrated_passed,
            "backend_checks_passed": (
                calibrated_passed
                if acceptance_policy == CALIBRATED_ACCEPTANCE_POLICY
                else legacy_passed
            ),
        }
        candidate["accepted"] = bool(
            candidate["primary_timings_stable"] and candidate["backend_checks_passed"]
        )
        candidates[packed_arm] = candidate
    return candidates


def exact_trace(actual, expected):
    if actual["cache_length"] != expected["cache_length"]:
        raise RuntimeError("cache length drift")
    for key in ("logits", "predictions", "fed_tokens"):
        if not torch.equal(actual[key], expected[key]):
            raise RuntimeError(f"repeat/Graph equivalence failed: {key}")


def require_routes(routes, steps):
    if len(routes) != 252 or any(
        route != {"dense_fallback": 0, "packed_m1": steps} for route in routes.values()
    ):
        raise RuntimeError("prepared W3 decode route coverage failed")


def compact(trace):
    return {
        "logits_sha256": tensor_sha256(trace["logits"]),
        "predictions": trace["predictions"].tolist(),
        "fed_tokens": trace["fed_tokens"].tolist(),
        "cache_length": trace["cache_length"],
        "routes": trace["routes"],
    }


def validate_protocol(config):
    expected_tiles = {f"{out}x{inner}": tile for (out, inner), tile in QWEN3_ROW_TILE_BY_SHAPE.items()}
    corrected_arms = [
        "original_bf16",
        "decoded_w3_bf16",
        "packed_w3_fast_corrected",
        "packed_w3_observed_exact",
    ]
    packed_routes = config.get("packed_routes", {"packed_w3_inline": "structural"})
    if (
        config.get("arms")
        not in (["original_bf16", "decoded_w3_bf16", "packed_w3_inline"], corrected_arms)
        or config.get("modes") != ["prepared_eager", "sequence_graph"]
        or config.get("primary_mode") != "sequence_graph"
        or config.get("row_tile_by_shape") != expected_tiles
        or config.get("prepare_candidate") is not False
        or config.get("decode_steps") != 32
        or config.get("warmup", 0) < 1
        or config.get("repeats", 0) < 3
    ):
        raise ValueError("frozen W3 full-model protocol drifted")
    acceptance_policy = config.get("backend_acceptance_policy", LEGACY_ACCEPTANCE_POLICY)
    if acceptance_policy not in {LEGACY_ACCEPTANCE_POLICY, CALIBRATED_ACCEPTANCE_POLICY}:
        raise ValueError("unknown backend_acceptance_policy")
    trim = config.get("timing_trim_per_side", 0)
    limit = config.get("packed_vs_decoded_relative_limit", 1.0)
    logprob_relative_limit = config.get("packed_vs_decoded_logprob_relative_limit")
    if not isinstance(trim, int) or isinstance(trim, bool) or not 0 <= trim <= 2:
        raise ValueError("timing_trim_per_side must be an integer in [0,2]")
    # A declared trim must actually apply, otherwise statistics_row silently
    # falls back to the raw range and the protocol would misdescribe the gate.
    if trim and config["repeats"] - 2 * trim < 3:
        raise ValueError("repeats too small for the declared timing_trim_per_side")
    if (
        not isinstance(limit, (int, float))
        or isinstance(limit, bool)
        or not math.isfinite(limit)
        or not limit > 0
    ):
        raise ValueError("packed_vs_decoded_relative_limit must be a positive number")
    if acceptance_policy == CALIBRATED_ACCEPTANCE_POLICY:
        if (
            not isinstance(logprob_relative_limit, (int, float))
            or isinstance(logprob_relative_limit, bool)
            or not math.isfinite(logprob_relative_limit)
            or not logprob_relative_limit > 0
        ):
            raise ValueError(
                "packed_vs_decoded_logprob_relative_limit must be a positive number"
            )
    elif logprob_relative_limit is not None:
        raise ValueError("legacy protocol must not redefine calibrated logprob acceptance")
    fused = config.get("fused_nonlinear_modules", {"rms_norm": False, "rope": False})
    if not isinstance(fused, dict) or set(fused) != {"rms_norm", "rope"} or not all(
        isinstance(value, bool) for value in fused.values()
    ):
        raise ValueError("fused_nonlinear_modules must declare bool rms_norm and rope")
    expected_routes = (
        {"packed_w3_inline": "structural"}
        if config["arms"][-1] == "packed_w3_inline"
        else {
            "packed_w3_fast_corrected": "fast_corrected",
            "packed_w3_observed_exact": "observed_exact",
        }
    )
    if packed_routes != expected_routes:
        raise ValueError("frozen W3 packed-route mapping drifted")
    for route in packed_routes.values():
        if route not in QWEN3_W3_ROUTE_POLICIES:
            raise ValueError(f"unknown packed route: {route}")
    if config["arms"] == corrected_arms and (
        config.get("target_gpu_family") != "NVIDIA A100"
        or config.get("target_compute_capability") != [8, 0]
        or config.get("target_min_vram_bytes") != 75000000000
        or config.get("linear_evidence_sha256")
        != "4f567a9adf28d80eb2f7d08f08146a08a855d80f3c9589f02d5bd845f51460ca"
    ):
        raise ValueError("corrected full-model provenance/device gate drifted")


def packed_routes(config):
    return config.get("packed_routes", {"packed_w3_inline": "structural"})


def load_model(snapshot, layout_root, manifest_sha256, arm, route_by_arm):
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    torch.cuda.synchronize()
    coverage = None
    if arm == "decoded_w3_bf16":
        for layer, decoder in enumerate(model.model.layers):
            payload, _ = load_w3_lut_layer(
                layout_root, layer, expected_manifest_sha256=manifest_sha256
            )
            for name in QWEN3_LINEAR_MODULES:
                layout = {field: payload[f"{name}.{field}"].cuda() for field in FIELDS}
                weight = restore_planar_w3(layout, dtype=torch.bfloat16)
                decoder.get_submodule(name).weight.copy_(weight)
                del layout, weight
            del payload
    elif arm in route_by_arm:
        coverage = replace_w3_model_linears(
            model,
            layout_root,
            expected_manifest_sha256=manifest_sha256,
            allow_prefill_fallback=True,
            route=route_by_arm[arm],
        )
        if coverage["linear_count"] != 252:
            raise RuntimeError("incomplete packed W3 coverage")
    torch.cuda.synchronize()
    return model, coverage


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CALIBRATED_PROTOCOL)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA required; no CPU/MPS substitution")

    protocol_path = args.protocol.resolve()
    if protocol_path not in {
        PROTOCOL.resolve(),
        CORRECTED_PROTOCOL.resolve(),
        CALIBRATED_PROTOCOL.resolve(),
        FUSED_CALIBRATED_PROTOCOL.resolve(),
    }:
        raise ValueError("protocol must be a repository-frozen W3 full-model config")
    config = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    acceptance_policy = config.get("backend_acceptance_policy", LEGACY_ACCEPTANCE_POLICY)
    route_by_arm = packed_routes(config)
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sources = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }
    if environment.get("status") != "ready_for_gpu_trial" or environment.get("source_sha256") != sources:
        raise ValueError("capture a fresh matching environment/source record")
    if (
        environment.get("torch", {}).get("version") != torch.__version__
        or environment.get("torch", {}).get("cuda_runtime") != torch.version.cuda
        or environment.get("torch", {}).get("devices", [{}])[0].get("name")
        != torch.cuda.get_device_name(0)
    ):
        raise ValueError("runtime differs from environment record")
    if "target_gpu_family" in config and (
        config["target_gpu_family"] not in torch.cuda.get_device_name(0)
        or list(torch.cuda.get_device_capability(0)) != config["target_compute_capability"]
        or torch.cuda.get_device_properties(0).total_memory < config["target_min_vram_bytes"]
    ):
        raise ValueError("corrected trial requires the frozen A100 target")
    if sha256_file(args.w3_layout_root / "manifest.json") != args.w3_layout_manifest_sha256:
        raise ValueError("W3 layout manifest drifted")

    pinned = json.loads(
        (ROOT / "configs/evaluation/qwen3_8b_wikitext2_distilled_step400_v1.json").read_text()
    )
    if args.snapshot_root.name != pinned["model"]["revision"]:
        raise ValueError("snapshot revision drift")
    for name, digest in pinned["model_preflight_files"].items():
        if sha256_file(args.snapshot_root / name) != digest:
            raise ValueError(f"snapshot hash drift: {name}")
    payloads = [
        load_w3_lut_layer(
            args.w3_layout_root,
            layer,
            expected_manifest_sha256=args.w3_layout_manifest_sha256,
        )[1]
        for layer in range(36)
    ]
    for row_tile in sorted(set(QWEN3_ROW_TILE_BY_SHAPE.values())):
        load_w3_lut_extension(row_tile)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot_root, local_files_only=True)
    prompts = [tokenizer(text, return_tensors="pt").input_ids.cuda() for text in config["prompts"]]
    if any(not 2 <= prompt.shape[1] <= config["max_prompt_tokens"] for prompt in prompts):
        raise ValueError("prompt length drift")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    report = {
        "status": "running",
        "stage": "w3_inline_prepared_static_full_model_m1",
        "protocol": config,
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "environment_sha256": sha256_file(args.environment),
        "source_sha256": sources,
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "payloads": payloads,
        "prompt_token_sha256": [tensor_sha256(prompt) for prompt in prompts],
        "primary_baseline": "original_bf16",
        "primary_mode": config["primary_mode"],
        "primary_metric": "same-run full-sequence CUDA Graph total",
        "numerical_policy": (
            "packed-vs-decoded W3 uses same-run relative NRMSE and max-logprob budgets plus finite/shape/token invariants; the legacy absolute 0.005/0.05 gates and quantized-vs-original comparison are report-only; prepared wrapper and Graph are exact"
            if acceptance_policy == CALIBRATED_ACCEPTANCE_POLICY
            else "historical packed-vs-decoded W3 absolute 0.005 NRMSE plus 0.05 max-logprob gate; calibrated comparisons are diagnostic only; quantized-vs-original is report-only; prepared wrapper and Graph are exact"
        ),
        "backend_acceptance_policy": {
            "id": acceptance_policy,
            "nrmse_scope": "complete prefill-plus-32-decode-position logits trace",
            "control": "same-run same-prompt original_bf16 dynamic-vs-static trace NRMSE",
            "relative_limit": config.get("packed_vs_decoded_relative_limit", 1.0),
            "logprob_control": "same-run same-prompt original_bf16 dynamic-vs-static max log-probability error",
            "logprob_relative_limit": config.get(
                "packed_vs_decoded_logprob_relative_limit", 1.0
            ),
            "required_invariants": [
                "shape_and_finite_logit_metrics",
                "fed_tokens_equal",
                "greedy_tokens_equal",
                "finite_logprob_metric",
            ],
            "legacy_absolute_0.005_nrmse_and_0.05_logprob_gates": (
                "report_only"
                if acceptance_policy == CALIBRATED_ACCEPTANCE_POLICY
                else "acceptance"
            ),
        },
        "scope": "batch1 real-prefix static KV, 32 fixed continuation tokens, embedding/all36blocks/LM head/argmax",
        "excluded_from_decode": "load, conversion, prefill, KV reset, graph capture, audits, CPU output copies",
        "residency_policy": "all configured models and prompt caches resident; interleaved arm order",
        "prepare_candidate": False,
        "arms": {},
        "execution_order": [],
        "next_stage": "not_launched",
    }
    atomic_json(args.output, report)
    # Patching happens before any model runs and covers every arm, so the arms
    # keep sharing one non-Linear path. The fused kernels are not bit-exact with
    # stock, so a partial application would silently invalidate the comparison.
    fusion = ExitStack()
    try:
        requested = config.get(
            "fused_nonlinear_modules", {"rms_norm": False, "rope": False}
        )
        report["fused_nonlinear_modules"] = (
            fusion.enter_context(fused_qwen3_modules(**requested))
            if any(requested.values())
            else dict(requested)
        )
        models = {}
        sessions = {}
        dynamic = {}
        checked_static = {}
        audits = {}
        references = {}
        for arm in config["arms"]:
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            start = time.perf_counter()
            before = torch.cuda.memory_allocated()
            model, coverage = load_model(
                args.snapshot_root,
                args.w3_layout_root,
                args.w3_layout_manifest_sha256,
                arm,
                route_by_arm,
            )
            models[arm] = model
            arm_record = {
                "load_conversion_seconds": time.perf_counter() - start,
                "coverage": coverage,
                "resident_increment_bytes": torch.cuda.memory_allocated() - before,
                "prompts": [],
            }
            report["arms"][arm] = arm_record
            for prompt_index, input_ids in enumerate(prompts):
                original = references.get(prompt_index)
                trace = decode_trace(
                    model,
                    input_ids,
                    steps=config["decode_steps"],
                    forced_tokens=None if original is None else original["fed_tokens"],
                )
                if not torch.isfinite(trace["logits"]).all():
                    raise RuntimeError("nonfinite dynamic oracle")
                if arm == "original_bf16":
                    references[prompt_index] = trace
                if arm in route_by_arm and not validate_routes(
                    trace, expected_linears=252, steps=config["decode_steps"]
                ):
                    raise RuntimeError("dynamic packed W3 coverage failed")
                dynamic[arm, prompt_index] = trace
                start = time.perf_counter()
                sessions[arm, prompt_index] = StaticDecodeSession(
                    model, input_ids, references[prompt_index]["fed_tokens"]
                )
                torch.cuda.synchronize()
                checked_static[arm, prompt_index] = sessions[arm, prompt_index].audit()
                arm_record["prompts"].append(
                    {
                        "prompt_index": prompt_index,
                        "dynamic_audit": compact(trace),
                        "timings": {},
                        "static_prefix_setup_seconds": time.perf_counter() - start,
                    }
                )

        with ExitStack() as stack:
            for model in models.values():
                stack.enter_context(prepared_linears(model, torch.bfloat16))
            for (arm, prompt_index), session in sessions.items():
                audit = session.audit()
                audits[arm, prompt_index] = audit
                dynamic_static = compare_trace(
                    audit,
                    dynamic[arm, prompt_index],
                    logprob_tolerance=config["logprob_max_abs_tolerance"],
                )
                exact_trace(audit, checked_static[arm, prompt_index])
                if not dynamic_static["fed_tokens_equal"]:
                    raise RuntimeError("dynamic/static fed-token drift")
                if arm in route_by_arm:
                    require_routes(audit["routes"], config["decode_steps"])
                graph_trace = session.capture()
                exact_trace(graph_trace, audit)
                if arm in route_by_arm:
                    require_routes(session.capture_routes, config["decode_steps"])
                report["arms"][arm]["prompts"][prompt_index].update(
                    prepared_audit=compact(audit),
                    dynamic_static_check=dynamic_static,
                    checked_static_audit=compact(checked_static[arm, prompt_index]),
                    prepared_wrapper_exact=True,
                    graph_exact=True,
                    capture_routes=session.capture_routes,
                )

            report["all_models_caches_graphs_allocated_bytes"] = torch.cuda.memory_allocated()
            report["setup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            for _ in range(config["warmup"]):
                for session in sessions.values():
                    for mode in config["modes"]:
                        session.measure(mode)
            samples = {
                (arm, prompt_index, mode, key): []
                for arm, prompt_index in sessions
                for mode in config["modes"]
                for key in ("wall_ms", "device_ms")
            }
            for repeat in range(config["repeats"]):
                offset = repeat % len(config["arms"])
                arms = config["arms"][offset:] + config["arms"][:offset]
                if repeat % 2:
                    arms = list(reversed(arms))
                modes = config["modes"] if repeat % 2 == 0 else list(reversed(config["modes"]))
                for prompt_index in range(len(prompts)):
                    for mode in modes:
                        for arm in arms:
                            timing, trace = sessions[arm, prompt_index].measure(mode)
                            exact_trace(trace, audits[arm, prompt_index])
                            report["execution_order"].append(
                                {
                                    "repeat": repeat,
                                    "prompt": prompt_index,
                                    "mode": mode,
                                    "arm": arm,
                                }
                            )
                            for key, value in timing.items():
                                samples[arm, prompt_index, mode, key].append(value)
                for arm, prompt_index in sessions:
                    report["arms"][arm]["prompts"][prompt_index]["timings"] = {
                        mode: {
                            key: statistics_row(
                                samples[arm, prompt_index, mode, key],
                                config["max_relative_timing_range"],
                                trim_per_side=config.get("timing_trim_per_side", 0),
                            )
                            for key in ("wall_ms", "device_ms")
                        }
                        for mode in config["modes"]
                    }
                atomic_json(args.output, report)

        comparisons = []
        for prompt_index in range(len(prompts)):
            quantization_report = compare_trace(
                audits["decoded_w3_bf16", prompt_index],
                audits["original_bf16", prompt_index],
                logprob_tolerance=config["logprob_max_abs_tolerance"],
            )
            control = report["arms"]["original_bf16"]["prompts"][prompt_index]
            control_check = control["dynamic_static_check"]
            control_nrmse = control_check["logits"]["normalized_rmse"]
            control_logprob_max_error = control_check["max_abs_logprob_error"]
            for mode in config["modes"]:
                rows = {
                    arm: report["arms"][arm]["prompts"][prompt_index]["timings"][mode]
                    for arm in config["arms"]
                }
                for packed_arm, route in route_by_arm.items():
                    compared_arms = ("original_bf16", "decoded_w3_bf16", packed_arm)
                    stable = all(
                        value["stable"]
                        for arm in compared_arms
                        for value in rows[arm].values()
                    )
                    backend_check = compare_trace(
                        audits[packed_arm, prompt_index],
                        audits["decoded_w3_bf16", prompt_index],
                        logprob_tolerance=config["logprob_max_abs_tolerance"],
                    )
                    calibrated_check = calibrated_backend_gate(
                        backend_check,
                        control_nrmse,
                        config.get("packed_vs_decoded_relative_limit", 1.0),
                        control_logprob_max_error,
                        config.get("packed_vs_decoded_logprob_relative_limit", 1.0),
                    )
                    packed = rows[packed_arm]["wall_ms"]["median"]
                    comparisons.append(
                        {
                            "prompt_index": prompt_index,
                            "mode": mode,
                            "packed_arm": packed_arm,
                            "route": route,
                            "primary": mode == config["primary_mode"],
                            "stable": stable,
                            "speedup_vs_original": (
                                rows["original_bf16"]["wall_ms"]["median"] / packed
                                if stable
                                else None
                            ),
                            "speedup_vs_decoded_w3": (
                                rows["decoded_w3_bf16"]["wall_ms"]["median"] / packed
                                if stable
                                else None
                            ),
                            "packed_vs_decoded_w3_check": backend_check,
                            "packed_vs_decoded_relative_check": calibrated_check["relative_check"],
                            "packed_vs_decoded_logprob_relative_check": calibrated_check[
                                "relative_logprob_check"
                            ],
                            "packed_vs_decoded_backend_acceptance": calibrated_check,
                            "decoded_w3_vs_original_report_only": quantization_report,
                        }
                    )
        report["comparisons"] = comparisons
        primary = [item for item in comparisons if item["primary"]]
        report["all_timings_stable"] = all(item["stable"] for item in comparisons)
        report["primary_timings_stable"] = all(item["stable"] for item in primary)
        report["all_legacy_absolute_backend_checks_passed"] = all(
            item["packed_vs_decoded_w3_check"]["passed"] for item in primary
        )
        report["all_relative_nrmse_checks_passed"] = all(
            item["packed_vs_decoded_relative_check"]["passed"] for item in primary
        )
        report["all_relative_logprob_checks_passed"] = all(
            item["packed_vs_decoded_logprob_relative_check"]["passed"]
            for item in primary
        )
        report["all_relative_backend_checks_passed"] = all(
            item["packed_vs_decoded_backend_acceptance"]["relative_checks_passed"]
            for item in primary
        )
        report["all_backend_invariants_passed"] = all(
            item["packed_vs_decoded_backend_acceptance"]["invariants_passed"]
            for item in primary
        )
        calibrated_checks_passed = all(
            item["packed_vs_decoded_backend_acceptance"]["passed"] for item in primary
        )
        report["all_calibrated_backend_checks_passed"] = calibrated_checks_passed
        report["all_backend_checks_passed"] = (
            calibrated_checks_passed
            if acceptance_policy == CALIBRATED_ACCEPTANCE_POLICY
            else report["all_legacy_absolute_backend_checks_passed"]
        )
        report["all_dynamic_static_checks_passed"] = all(
            prompt["dynamic_static_check"]["passed"]
            for arm in report["arms"].values()
            for prompt in arm["prompts"]
        )
        report["candidate_acceptance"] = build_candidate_acceptance(
            primary, route_by_arm, acceptance_policy
        )
        if not report["all_backend_checks_passed"]:
            report["status"] = "completed_with_backend_numerical_differences"
        elif not report["primary_timings_stable"]:
            report["status"] = "completed_with_unstable_primary_graph"
        else:
            report["status"] = "completed_pending_review"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        fusion.close()
        atomic_json(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
