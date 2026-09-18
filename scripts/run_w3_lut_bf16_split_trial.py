#!/usr/bin/env python3
"""Run the bounded W3 BF16-rounding correction and split-G Linear trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing
from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256
from fluxbin_style.gptq_deployment import (
    FIELDS,
    bf16_weight_semantics_w3_matvec,
    structural_w3_matvec,
)
from fluxbin_style.w3_lut_artifacts import load_w3_lut_layer
from fluxbin_style.w3_lut_deployment import (
    ARITHMETIC_MODES,
    EXPERIMENTAL_ROW_TILES,
    finish,
    inline_main,
    load_w3_lut_extension,
    w3_lut_m1_out,
    workspace_shape,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/acceleration/w3_lut_bf16_split_candidates_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--gptq-root", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }


def candidate_count(config: dict) -> int:
    modes = len(config.get("arithmetic_modes", []))
    return modes * sum(
        len(record.get("row_tiles", [])) * len(record.get("groups_per_split", []))
        for record in config.get("shape_classes", {}).values()
    )


def validate_config(config: dict) -> None:
    expected = {
        "q_o": ("self_attn.q_proj", [4096, 4096]),
        "k_v": ("self_attn.k_proj", [1024, 4096]),
        "gate_up": ("mlp.gate_proj", [12288, 4096]),
        "down": ("mlp.down_proj", [4096, 12288]),
    }
    if (
        config.get("dtype") != "bfloat16"
        or config.get("group_size") != 128
        or config.get("mode") != "cuda_graph_total"
        or tuple(config.get("arithmetic_modes", ())) != ARITHMETIC_MODES
        or set(config.get("shape_classes", {})) != set(expected)
        or config.get("candidate_count") != candidate_count(config)
        or config.get("acceptance_boundary")
        != "diagnostic only; no production dispatch or full-model promotion without separate correctness and full-model gates"
    ):
        raise ValueError("W3 BF16/split diagnostic contract drifted")
    for shape_class, (module, shape) in expected.items():
        record = config["shape_classes"][shape_class]
        if record.get("module") != module or record.get("shape") != shape:
            raise ValueError(f"shape contract drifted: {shape_class}")
        if not record.get("row_tiles") or not set(record["row_tiles"]).issubset(EXPERIMENTAL_ROW_TILES):
            raise ValueError(f"invalid row tiles: {shape_class}")
        groups = shape[1] // 128
        if not record.get("groups_per_split") or any(
            not isinstance(value, int) or not 1 <= value <= groups
            for value in record["groups_per_split"]
        ):
            raise ValueError(f"invalid groups_per_split: {shape_class}")
    if config.get("warmup", 0) < 1 or config.get("repeats", 0) < 1 or config.get("rounds", 0) < 3:
        raise ValueError("invalid timing repetition contract")


def capture(functions: dict[str, object]):
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
        raise RuntimeError("W3 BF16/split trial requires NVIDIA CUDA")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sources = source_hashes()
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

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tiles = sorted(
        {
            tile
            for record in config["shape_classes"].values()
            for tile in record["row_tiles"]
        }
    )
    for tile in tiles:
        load_w3_lut_extension(tile)

    layer = int(config["layer"])
    tensors, entry = load_w3_lut_layer(
        args.w3_layout_root,
        layer,
        expected_manifest_sha256=args.w3_layout_manifest_sha256,
    )
    decoded_path = args.gptq_root / "decoded_bf16" / f"layer-{layer:03d}.safetensors"
    report = {
        "status": "running",
        "stage": "w3_lut_bf16_rounding_and_split_g",
        "diagnostic_only": True,
        "config_sha256": sha256_file(args.config),
        "environment_sha256": sha256_file(args.environment),
        "source_sha256": sources,
        "runner_sha256": sha256_file(Path(__file__)),
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "w3_layout_layer": entry,
        "gpu": torch.cuda.get_device_name(0),
        "metric": "same-run CUDA Graph total per call",
        "cells": [],
        "stage_timings": [],
    }
    atomic_json(args.output, report)
    try:
        with safe_open(decoded_path, framework="pt", device="cpu") as decoded_handle:
            for shape_class, spec in config["shape_classes"].items():
                module = spec["module"]
                layout = {field: tensors[f"{module}.{field}"].cuda() for field in FIELDS}
                decoded = decoded_handle.get_tensor(f"{module}.weight").cuda()
                out_features, in_features = decoded.shape
                if decoded.dtype != torch.bfloat16 or [out_features, in_features] != spec["shape"]:
                    raise ValueError(f"decoded weight contract drifted: {shape_class}")
                generator = torch.Generator(device="cuda").manual_seed(config["seed"])
                x = torch.randn(
                    1,
                    in_features,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                dense_out = torch.empty(1, out_features, device="cuda", dtype=torch.bfloat16)

                def decoded_dense():
                    return torch.mm(x, decoded.t(), out=dense_out)

                structural_reference = structural_w3_matvec(x, layout).bfloat16()
                corrected_reference = bf16_weight_semantics_w3_matvec(x, layout).bfloat16()
                decoded_dense()
                reference_checks = {
                    "structural_vs_decoded": numerical_gate(structural_reference, dense_out),
                    "corrected_vs_decoded": numerical_gate(corrected_reference, dense_out),
                }
                stage_row_tile = spec["row_tiles"][0]
                for arithmetic in config["arithmetic_modes"]:
                    stage_out = torch.empty_like(dense_out)
                    stage_workspace = torch.full(
                        workspace_shape(out_features, in_features),
                        float("nan"),
                        device="cuda",
                    )

                    def main_stage():
                        inline_main(
                            x,
                            layout,
                            stage_workspace,
                            row_tile=stage_row_tile,
                            arithmetic=arithmetic,
                        )

                    def finish_stage():
                        finish(stage_workspace, stage_out, row_tile=stage_row_tile)

                    main_stage()
                    finish_stage()
                    stage_expected = (
                        structural_reference
                        if arithmetic == "structural"
                        else corrected_reference
                    )
                    stage_check = numerical_gate(stage_out, stage_expected)
                    if not stage_check["passed"] or not bool(torch.isfinite(stage_workspace).all()):
                        raise RuntimeError(
                            f"stage correctness failed: {shape_class}/R{stage_row_tile}/{arithmetic}"
                        )
                    stage_functions, stage_graphs = capture(
                        {"main": main_stage, "finish": finish_stage}
                    )
                    stage_graph_check = numerical_gate(stage_out, stage_expected)
                    if not stage_graph_check["passed"]:
                        raise RuntimeError(
                            f"stage graph correctness failed: {shape_class}/R{stage_row_tile}/{arithmetic}"
                        )
                    report["stage_timings"].append(
                        {
                            "shape_class": shape_class,
                            "module": module,
                            "shape": spec["shape"],
                            "row_tile": stage_row_tile,
                            "groups_per_split": 1,
                            "arithmetic": arithmetic,
                            "correctness": stage_check,
                            "cuda_graph_correctness": stage_graph_check,
                            "timings": paired_cuda_timing(
                                stage_functions,
                                warmup=int(config["warmup"]),
                                repeats=int(config["repeats"]),
                                rounds=int(config["rounds"]),
                            ),
                            "warning": "main and finish were captured and timed separately; durations are diagnostic and non-additive",
                        }
                    )
                    atomic_json(args.output, report)
                    del stage_functions, stage_graphs, stage_out, stage_workspace
                for row_tile in spec["row_tiles"]:
                    for groups_per_split in spec["groups_per_split"]:
                        for arithmetic in config["arithmetic_modes"]:
                            candidate_out = torch.empty_like(dense_out)
                            workspace = torch.full(
                                workspace_shape(
                                    out_features,
                                    in_features,
                                    groups_per_split,
                                ),
                                float("nan"),
                                device="cuda",
                            )

                            def candidate():
                                return w3_lut_m1_out(
                                    x,
                                    layout,
                                    candidate_out,
                                    workspace,
                                    row_tile=row_tile,
                                    groups_per_split=groups_per_split,
                                    arithmetic=arithmetic,
                                )

                            candidate()
                            first = candidate_out.clone()
                            expected = (
                                structural_reference
                                if arithmetic == "structural"
                                else corrected_reference
                            )
                            correctness = numerical_gate(candidate_out, expected)
                            correctness.update(
                                repeat_exact=False,
                                workspace_finite=(
                                    workspace.shape[0] == 1
                                    or bool(torch.isfinite(workspace).all())
                                ),
                                workspace_written=bool(torch.isfinite(workspace).all()),
                                error_vs_decoded_bf16=numerical_gate(candidate_out, dense_out),
                                error_vs_matching_reference=numerical_gate(candidate_out, expected),
                            )
                            workspace.fill_(float("nan"))
                            candidate()
                            correctness["repeat_exact"] = bool(torch.equal(first, candidate_out))
                            if not all(
                                (
                                    correctness["passed"],
                                    correctness["repeat_exact"],
                                    correctness["workspace_finite"],
                                )
                            ):
                                raise RuntimeError(
                                    f"correctness failed: {shape_class}/R{row_tile}/gps{groups_per_split}/{arithmetic}"
                                )
                            functions, graphs = capture(
                                {"decoded_w3_bf16": decoded_dense, "candidate": candidate}
                            )
                            graph_correctness = numerical_gate(candidate_out, expected)
                            correctness["cuda_graph_matching_reference"] = graph_correctness
                            if not graph_correctness["passed"]:
                                raise RuntimeError(
                                    f"CUDA Graph correctness failed: {shape_class}/R{row_tile}/gps{groups_per_split}/{arithmetic}"
                                )
                            timings = paired_cuda_timing(
                                functions,
                                warmup=int(config["warmup"]),
                                repeats=int(config["repeats"]),
                                rounds=int(config["rounds"]),
                            )
                            stable = all(
                                item["relative_range"] <= config["maximum_relative_range"]
                                for item in timings.values()
                            )
                            candidate_us = timings["candidate"]["median_us"]
                            report["cells"].append(
                                {
                                    "shape_class": shape_class,
                                    "module": module,
                                    "shape": spec["shape"],
                                    "row_tile": row_tile,
                                    "groups_per_split": groups_per_split,
                                    "splits": workspace.shape[0],
                                    "arithmetic": arithmetic,
                                    "input_sha256": tensor_sha256(x),
                                    "reference_checks": reference_checks,
                                    "correctness": correctness,
                                    "timings": timings,
                                    "timing_stable": stable,
                                    "speedup_vs_decoded_w3_bf16": (
                                        timings["decoded_w3_bf16"]["median_us"] / candidate_us
                                        if stable
                                        else None
                                    ),
                                }
                            )
                            atomic_json(args.output, report)
                            del candidate_out, first, functions, graphs, workspace
                del corrected_reference, decoded, dense_out, layout, structural_reference, x
        if len(report["cells"]) != config["candidate_count"]:
            raise RuntimeError("candidate coverage drifted")
        report["status"] = (
            "completed_pending_review"
            if all(cell["timing_stable"] for cell in report["cells"])
            else "completed_with_unstable_cells"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
