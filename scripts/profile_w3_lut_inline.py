#!/usr/bin/env python3
"""Capture bounded Nsight Compute diagnostics for one accepted W3 shape winner.

This runner deliberately emits no performance ratios.  Formal latency remains
the same-run CUDA Graph result produced by ``run_w3_lut_benchmark.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fluxbin_style.acceleration_checks import numerical_gate
from fluxbin_style.deployment import (
    FIELDS as QBB_FIELDS,
    convert_artifact as convert_qbb,
    decode_layout as decode_qbb,
    load_extension as load_qbb_extension,
    m1_out as qbb_m1_out,
    workspace_shape as qbb_workspace_shape,
)
from fluxbin_style.deployment_artifacts import load_accepted_layer
from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256
from fluxbin_style.gptq_deployment import FIELDS, structural_w3_matvec
from fluxbin_style.w3_lut_artifacts import load_w3_lut_layer
from fluxbin_style.w3_lut_deployment import (
    ROW_TILES,
    load_w3_lut_extension,
    w3_lut_m1_out,
    workspace_shape as w3_workspace_shape,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/acceleration/w3_lut_candidates_v1.json"
SHAPE_CLASSES = ("q_o", "k_v", "gate_up", "down")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--qbb-artifact-root", type=Path, required=True)
    parser.add_argument("--shape-class", choices=SHAPE_CLASSES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }


def select_winner(report: dict, shape_class: str) -> dict:
    candidates = [cell for cell in report.get("cells", []) if cell.get("shape_class") == shape_class]
    observed_tiles = sorted(cell.get("row_tile") for cell in candidates)
    if observed_tiles != list(ROW_TILES):
        raise ValueError(f"incomplete row-tile coverage for {shape_class}: {observed_tiles}")
    accepted = [
        cell
        for cell in candidates
        if cell.get("timing_stable") is True
        and cell.get("correctness", {}).get("passed") is True
        and cell.get("correctness", {}).get("repeat_exact") is True
        and cell.get("correctness", {}).get("workspace_finite") is True
    ]
    if not accepted:
        raise ValueError(f"no stable, correct inline candidate for {shape_class}")
    return min(accepted, key=lambda cell: cell["timings"]["w3_lut_inline"]["median_us"])


def validate_inputs(config: dict, environment: dict, benchmark: dict, sources: dict) -> None:
    if environment.get("status") != "ready_for_gpu_trial":
        raise ValueError("environment is not GPU-ready")
    if environment.get("source_sha256") != sources:
        raise ValueError("environment/source hashes drifted")
    if environment.get("torch", {}).get("version") != torch.__version__:
        raise ValueError("PyTorch version drifted")
    if environment.get("torch", {}).get("cuda_runtime") != torch.version.cuda:
        raise ValueError("CUDA runtime drifted")
    if environment.get("torch", {}).get("devices", [{}])[0].get("name") != torch.cuda.get_device_name(0):
        raise ValueError("GPU drifted")
    if benchmark.get("status") != "completed_pending_review":
        raise ValueError("a fully stable reviewed-input benchmark is required")
    if benchmark.get("metric") != "same-run CUDA Graph total per call":
        raise ValueError("formal benchmark metric drifted")
    if benchmark.get("prepare_in_round_one") is not False:
        raise ValueError("prepare policy drifted")
    if benchmark.get("source_sha256") != sources:
        raise ValueError("benchmark/source hashes drifted")
    if benchmark.get("config_sha256") != sha256_file(DEFAULT_CONFIG):
        raise ValueError("benchmark/config hash drifted")
    if len(benchmark.get("cells", [])) != config.get("candidate_count"):
        raise ValueError("benchmark cell count drifted")


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("W3 profiling requires NVIDIA CUDA; no CPU/MPS substitute")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark_result.read_text(encoding="utf-8"))
    sources = source_hashes()
    validate_inputs(config, environment, benchmark, sources)
    if benchmark.get("w3_layout_manifest_sha256") != args.w3_layout_manifest_sha256:
        raise ValueError("benchmark/W3 layout manifest hash drifted")

    winner = select_winner(benchmark, args.shape_class)
    module = config["shape_classes"][args.shape_class]
    if winner.get("module") != module or winner.get("shape") != config["expected_shapes"][args.shape_class]:
        raise ValueError("winner module/shape drifted")
    row_tile = int(winner["row_tile"])

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    load_w3_lut_extension(row_tile)
    load_qbb_extension("v5_p1024")

    layer = int(config["layer"])
    w3_tensors, w3_entry = load_w3_lut_layer(
        args.w3_layout_root,
        layer,
        expected_manifest_sha256=args.w3_layout_manifest_sha256,
    )
    qbb_tensors, qbb_entry = load_accepted_layer(args.qbb_artifact_root, layer)
    w3_layout = {field: w3_tensors[f"{module}.{field}"].cuda() for field in FIELDS}
    qbb_payload = {field: qbb_tensors[f"{module}.{field}"] for field in QBB_FIELDS}
    qbb_layout = {
        key: value.cuda()
        for key, value in convert_qbb(qbb_payload, kernel="v5_p1024").items()
    }

    out_features, in_features = winner["shape"]
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    x = torch.randn(1, in_features, generator=generator, device="cuda", dtype=torch.bfloat16)
    if tensor_sha256(x) != winner["input_sha256"]:
        raise ValueError("profile input differs from formal benchmark input")

    w3_out = torch.empty(1, out_features, device="cuda", dtype=torch.bfloat16)
    w3_workspace = torch.full(
        w3_workspace_shape(out_features, in_features), float("nan"), device="cuda"
    )
    qbb_out = torch.empty_like(w3_out)
    qbb_workspace = torch.full(
        qbb_workspace_shape(out_features, in_features, 1, kernel="v5_p1024"),
        float("nan"),
        device="cuda",
    )

    def run_w3():
        return w3_lut_m1_out(x, w3_layout, w3_out, w3_workspace, row_tile=row_tile)

    def run_v5():
        return qbb_m1_out(
            x,
            qbb_layout,
            qbb_out,
            qbb_workspace,
            groups_per_split=1,
            kernel="v5_p1024",
        )

    report = {
        "status": "running",
        "diagnostic_only": True,
        "prepare_authorized": False,
        "shape_class": args.shape_class,
        "module": module,
        "shape": winner["shape"],
        "row_tile": row_tile,
        "winner_cuda_graph_total_us": winner["timings"]["w3_lut_inline"]["median_us"],
        "benchmark_result_sha256": sha256_file(args.benchmark_result),
        "environment_sha256": sha256_file(args.environment),
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "source_sha256": sources,
        "runner_sha256": sha256_file(Path(__file__)),
        "input_sha256": tensor_sha256(x),
        "w3_layout_layer": w3_entry,
        "qbb_layer": qbb_entry,
        "profile_calls_per_arm": 1,
        "scope": "one eager call each for accepted inline W3 and v5; counters only; no latency claim",
        "checks": {},
    }
    atomic_json(args.output, report)
    try:
        run_w3()
        w3_reference = structural_w3_matvec(x, w3_layout).to(torch.bfloat16)
        w3_check = numerical_gate(w3_out, w3_reference)
        w3_saved = w3_out.clone()
        w3_workspace.fill_(float("nan"))
        run_w3()
        w3_check.update(
            repeat_exact=bool(torch.equal(w3_saved, w3_out)),
            workspace_finite=bool(torch.isfinite(w3_workspace).all()),
        )

        qbb_dense = decode_qbb(qbb_layout, torch.bfloat16)
        qbb_reference = torch.mm(x, qbb_dense.t())
        run_v5()
        qbb_check = numerical_gate(qbb_out, qbb_reference)
        qbb_saved = qbb_out.clone()
        qbb_workspace.fill_(float("nan"))
        run_v5()
        qbb_check.update(
            repeat_exact=bool(torch.equal(qbb_saved, qbb_out)),
            workspace_finite=bool(torch.isfinite(qbb_workspace).all()),
        )
        report["checks"] = {"w3_lut_inline": w3_check, "v5_p1024_gps1": qbb_check}
        if not all(
            check["passed"] and check["repeat_exact"] and check["workspace_finite"]
            for check in report["checks"].values()
        ):
            raise RuntimeError("profile correctness/repeatability gate failed")

        for _ in range(20):
            run_w3()
            run_v5()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"w3_lut_inline_{args.shape_class}_r{row_tile}"):
                run_w3()
            with torch.cuda.nvtx.range(f"v5_p1024_gps1_{args.shape_class}"):
                run_v5()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        report["status"] = "capture_completed_pending_counter_review"
        report["timing_warning"] = "Profiler-instrumented execution is diagnostic only."
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
