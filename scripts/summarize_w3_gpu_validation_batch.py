#!/usr/bin/env python3
"""Summarize one rented-GPU validation batch into a single reviewable JSON.

Reads whatever the batch produced -- jobs are fail-soft, so missing or failed
results are normal and are reported as such rather than crashing the summary.
Computes nothing new: every number here is copied or divided from a result JSON,
so the summary can be regenerated from the archived evidence at any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-result", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=32)
    return parser.parse_args()


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "error": str(exc)}


def summarize_probe(report):
    if not report or report.get("status") != "completed":
        return {"available": False, "status": (report or {}).get("status")}
    return {
        "available": True,
        "timing_only": True,
        "all_cells_stable": report.get("all_cells_stable"),
        "cells": [
            {
                "shape_class": cell["shape_class"],
                "row_tile": cell["row_tile"],
                "real_us": cell["timings"]["real"]["median_us"],
                "uniform_index_us": cell["timings"]["uniform_index"]["median_us"],
                # Lower bound: the broadcast itself costs one shuffle per lookup.
                "conflict_cost_fraction_lower_bound": cell["conflict_cost_fraction_of_real"],
                "stable": cell["stable"],
            }
            for cell in report.get("cells", [])
        ],
    }


def summarize_sweep(report):
    if not report or report.get("status") != "completed":
        return {"available": False, "status": (report or {}).get("status")}
    cells = [cell for cell in report.get("cells", []) if "timings" in cell]
    best = {}
    unstable = 0
    for cell in cells:
        # Only stable cells may win: an unstable median is not a measurement.
        if not cell.get("timing_stable"):
            unstable += 1
            continue
        key = (cell.get("shape_class"), cell.get("arithmetic"))
        graph = cell["timings"].get("candidate", {}).get("median_us")
        if graph is None:
            continue
        if key not in best or graph < best[key]["median_us"]:
            best[key] = {
                "row_tile": cell.get("row_tile"),
                "groups_per_split": cell.get("groups_per_split"),
                "splits": cell.get("splits"),
                "median_us": graph,
                "speedup_vs_decoded_w3_bf16": cell.get("speedup_vs_decoded_w3_bf16"),
            }
    return {
        "available": True,
        "cell_count": len(cells),
        "unstable_cells_excluded": unstable,
        "best_per_shape_and_arithmetic": {
            f"{shape}/{arithmetic}": value for (shape, arithmetic), value in sorted(best.items())
        },
    }


def summarize_full_model(report, steps):
    if not report:
        return {"available": False, "status": None}
    arms = report.get("arms", {})
    rows = {}
    for arm, record in arms.items():
        for index, prompt in enumerate(record.get("prompts", [])):
            timing = prompt.get("timings", {}).get("sequence_graph", {}).get("device_ms")
            if not timing:
                continue
            rows[f"{arm}/p{index}"] = {
                "total_ms": timing["median"],
                "ms_per_token": timing["median"] / steps,
                "stable": timing.get("stable"),
                "relative_range": timing.get("relative_range"),
                "trimmed_relative_range": timing.get("trimmed_relative_range"),
            }
    return {
        "available": True,
        "status": report.get("status"),
        "fused_nonlinear_modules": report.get("fused_nonlinear_modules"),
        "backend_acceptance_policy": report.get("backend_acceptance_policy"),
        "candidate_acceptance": report.get("candidate_acceptance"),
        "primary_timings_stable": report.get("primary_timings_stable"),
        "all_backend_checks_passed": report.get("all_backend_checks_passed"),
        "all_calibrated_backend_checks_passed": report.get(
            "all_calibrated_backend_checks_passed"
        ),
        "all_backend_invariants_passed": report.get("all_backend_invariants_passed"),
        "all_relative_backend_checks_passed": report.get("all_relative_backend_checks_passed"),
        "all_relative_nrmse_checks_passed": report.get("all_relative_nrmse_checks_passed"),
        "all_relative_logprob_checks_passed": report.get(
            "all_relative_logprob_checks_passed"
        ),
        "all_legacy_absolute_backend_checks_passed": report.get(
            "all_legacy_absolute_backend_checks_passed"
        ),
        "sequence_graph_device_ms": rows,
        "primary_comparisons": [
            {
                "prompt_index": item["prompt_index"],
                "packed_arm": item.get("packed_arm"),
                "speedup_vs_original": item.get("speedup_vs_original"),
                "stable": item.get("stable"),
                "packed_vs_decoded_nrmse": item.get(
                    "packed_vs_decoded_w3_check", {}
                ).get("logits", {}).get("normalized_rmse"),
                "relative_check": item.get("packed_vs_decoded_relative_check", {}).get("ratio"),
                "relative_logprob_check": item.get(
                    "packed_vs_decoded_logprob_relative_check", {}
                ).get("ratio"),
                "backend_acceptance_passed": item.get(
                    "packed_vs_decoded_backend_acceptance", {}
                ).get("passed"),
                "backend_invariants_passed": item.get(
                    "packed_vs_decoded_backend_acceptance", {}
                ).get("invariants_passed"),
                "legacy_absolute_check_passed": item.get(
                    "packed_vs_decoded_backend_acceptance", {}
                ).get("legacy_absolute_check_passed"),
            }
            for item in report.get("comparisons", [])
            if item.get("primary")
        ],
    }


def fusion_delta(stock, fused, steps):
    """Same-session fused-vs-stock comparison, per arm and prompt.

    The fusions speed up every arm, so the honest headline is absolute
    milliseconds per token; the speedup ratio moves only because the non-Linear
    time is a shared additive term.
    """
    if not (stock.get("available") and fused.get("available")):
        return {"available": False}
    delta = {}
    for key, stock_row in stock["sequence_graph_device_ms"].items():
        fused_row = fused["sequence_graph_device_ms"].get(key)
        if not fused_row:
            continue
        delta[key] = {
            "stock_ms_per_token": stock_row["ms_per_token"],
            "fused_ms_per_token": fused_row["ms_per_token"],
            "saved_ms_per_token": stock_row["ms_per_token"] - fused_row["ms_per_token"],
            "fused_tokens_per_second": 1000.0 / fused_row["ms_per_token"],
            "stock_tokens_per_second": 1000.0 / stock_row["ms_per_token"],
        }
    return {"available": True, "per_arm_prompt": delta}


def main():
    args = parse_args()
    jobs = {}
    for status_file in sorted(args.job_dir.glob("*.status")):
        jobs[status_file.stem] = status_file.read_text(encoding="utf-8").strip()

    probe = summarize_probe(load(args.batch_result / "bank-conflict-probe/result.json"))
    sweep = summarize_sweep(load(args.batch_result / "split-sweep/result.json"))
    stock = summarize_full_model(
        load(args.batch_result / "full-model-stock/result.json"), args.decode_steps
    )
    fused = summarize_full_model(
        load(args.batch_result / "full-model-fused/result.json"), args.decode_steps
    )
    summary = {
        "jobs": jobs,
        "bank_conflict_probe": probe,
        "split_sweep": sweep,
        "full_model_stock": stock,
        "full_model_fused": fused,
        "fusion_delta": fusion_delta(stock, fused, args.decode_steps),
        "boundaries": [
            "the bank-conflict probe is timing-only; its build returns wrong values",
            "fused and stock full-model runs are different protocol ids and must not share a results table",
            "a job reported as failed produced no evidence; it is not a negative result",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
