#!/usr/bin/env python3
"""Measure the shared-memory bank-conflict cost of the W3 inline LUT lookup.

The inner loop reads `lut[chunk*256 + p]` where `p` is a weight byte, so the 32
lanes of a warp issue 32 data-dependent addresses into a 256-float table and
collide on banks (32 random balls in 32 bins: expected worst bin ~3.4). That is
the leading suspect for the kernel reaching only 19-52% of the effective weight
bandwidth cuBLAS BF16 achieves on the same shapes, but Nsight Compute is
unavailable on this host (ERR_NVGPUCTRPERM) so it cannot be confirmed with
hardware counters.

This probe answers it from wall time instead: it times the real kernel against
an otherwise identical build that broadcasts lane 0's index, which removes the
conflicts and keeps the weight loads live. The probe build returns WRONG values
by construction, so this script reports timing only, marks itself
diagnostic-only, and asserts that the probe output actually differs from the
real one rather than silently measuring the same kernel twice. The measured
delta is a lower bound on the conflict cost: the broadcast costs one shuffle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing
from fluxbin_style.evaluation import atomic_json, sha256_file
from fluxbin_style.gptq_deployment import FIELDS, structural_w3_matvec
from fluxbin_style.w3_lut_artifacts import load_w3_lut_layer
from fluxbin_style.w3_lut_deployment import (
    load_w3_lut_extension,
    workspace_shape,
)

ROOT = Path(__file__).resolve().parents[1]
# The frozen best row tile per shape class, from the accepted 4x3 Linear trial.
FROZEN_CELLS = (
    ("q_o", "self_attn.q_proj", (4096, 4096), 256),
    ("k_v", "self_attn.k_proj", (1024, 4096), 512),
    ("gate_up", "mlp.gate_proj", (12288, 4096), 1024),
    ("down", "mlp.down_proj", (4096, 12288), 1024),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--maximum-relative-range", type=float, default=0.10)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }


def capture(functions):
    for function in functions.values():
        for _ in range(3):
            function()
    torch.cuda.synchronize()
    graphs = []
    for function in functions.values():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graphs.append(graph)
    replay = {name: graph.replay for name, graph in zip(functions, graphs)}
    for function in replay.values():
        function()
    torch.cuda.synchronize()
    return replay, graphs


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("bank-conflict probe requires NVIDIA CUDA")
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sources = source_hashes()
    if environment.get("status") != "ready_for_gpu_trial":
        raise ValueError("environment is not GPU-ready")
    if environment.get("source_sha256") != sources:
        raise ValueError("environment/source hashes drifted")
    if environment.get("torch", {}).get("version") != torch.__version__:
        raise ValueError("PyTorch version drifted")
    if environment.get("torch", {}).get("devices", [{}])[0].get("name") != torch.cuda.get_device_name(0):
        raise ValueError("GPU drifted")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tensors, entry = load_w3_lut_layer(
        args.w3_layout_root,
        args.layer,
        expected_manifest_sha256=args.w3_layout_manifest_sha256,
    )
    report = {
        "status": "running",
        "stage": "w3_lut_shared_memory_bank_conflict_probe",
        "diagnostic_only": True,
        "timing_only": True,
        "probe_semantics": "uniform-index build returns wrong values by construction; never promote it",
        "delta_interpretation": "lower bound on conflict cost; the broadcast itself costs one shuffle per lookup",
        "environment_sha256": sha256_file(args.environment),
        "source_sha256": sources,
        "runner_sha256": sha256_file(Path(__file__)),
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "w3_layout_layer": entry,
        "gpu": torch.cuda.get_device_name(0),
        "metric": "same-run CUDA Graph total per call",
        "cells": [],
    }
    atomic_json(args.output, report)
    try:
        for shape_class, module, shape, row_tile in FROZEN_CELLS:
            layout = {field: tensors[f"{module}.{field}"].cuda() for field in FIELDS}
            out_features, in_features = shape
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            x = torch.randn(
                1, in_features, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            reference = structural_w3_matvec(x, layout).bfloat16()

            builds = {}
            for name, probe in (("real", False), ("uniform_index", True)):
                extension = load_w3_lut_extension(row_tile, uniform_index_probe=probe)
                out = torch.empty(1, out_features, device="cuda", dtype=torch.bfloat16)
                workspace = torch.full(
                    workspace_shape(out_features, in_features), float("nan"), device="cuda"
                )

                def run(extension=extension, out=out, workspace=workspace):
                    extension.inline_main(
                        x, layout["planes"], layout["scales"], layout["perm"], workspace
                    )
                    extension.finish(workspace, out)

                run()
                builds[name] = {"call": run, "out": out.clone(), "workspace": workspace}

            real_check = numerical_gate(builds["real"]["out"], reference)
            if not real_check["passed"]:
                raise RuntimeError(f"real build correctness failed: {shape_class}")
            # Guard against silently timing the same build twice, which would be
            # the failure mode that makes this probe look like a null result.
            if torch.equal(builds["real"]["out"], builds["uniform_index"]["out"]):
                raise RuntimeError(
                    f"probe build produced identical output: {shape_class}; "
                    "the uniform-index define did not take effect"
                )

            replay, graphs = capture({name: build["call"] for name, build in builds.items()})
            timings = paired_cuda_timing(
                replay,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
            del graphs
            real_us = timings["real"]["median_us"]
            probe_us = timings["uniform_index"]["median_us"]
            report["cells"].append(
                {
                    "shape_class": shape_class,
                    "module": module,
                    "shape": list(shape),
                    "row_tile": row_tile,
                    "real_correct": True,
                    "probe_output_differs": True,
                    "timings": timings,
                    "conflict_cost_fraction_of_real": (real_us - probe_us) / real_us,
                    "stable": all(
                        row["relative_range"] <= args.maximum_relative_range
                        for row in timings.values()
                    ),
                }
            )
            atomic_json(args.output, report)
        report["status"] = "completed"
        report["all_cells_stable"] = all(cell["stable"] for cell in report["cells"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
