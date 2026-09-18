#!/usr/bin/env python3
"""Run the frozen 4-shape x 3-row-tile inline W3 LUT CUDA-Graph trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing
from fluxbin_style.deployment import (
    FIELDS as QBB_FIELDS,
    convert_artifact as convert_qbb,
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
    workspace_shape,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/acceleration/w3_lut_candidates_v1.json"
W3_CONTRACT_FILES = (
    "configs/acceleration/w3_lut_candidates_v1.json",
    "scripts/record_acceleration_environment.py",
    "scripts/prepare_qwen3_8b_w3_lut_artifacts.py",
    "scripts/run_w3_lut_server_preflight.py",
    "scripts/run_w3_lut_benchmark.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--cuda-preflight", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--gptq-root", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--qbb-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_hashes():
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }


def contract_hashes():
    return {name: sha256_file(ROOT / name) for name in W3_CONTRACT_FILES}


def validate_config(config: dict) -> None:
    expected_shapes = {
        "q_o": "self_attn.q_proj",
        "k_v": "self_attn.k_proj",
        "gate_up": "mlp.gate_proj",
        "down": "mlp.down_proj",
    }
    expected_dimensions = {
        "q_o": [4096, 4096],
        "k_v": [1024, 4096],
        "gate_up": [12288, 4096],
        "down": [4096, 12288],
    }
    if (
        config.get("dtype") != "bfloat16"
        or config.get("scale_dtype") != "bfloat16"
        or config.get("group_size") != 128
        or config.get("groups_per_split") != 1
        or config.get("mode") != "cuda_graph_total"
        or config.get("row_tiles") != list(ROW_TILES)
        or config.get("shape_classes") != expected_shapes
        or config.get("expected_shapes") != expected_dimensions
        or config.get("candidate_count") != 12
        or config.get("prepare_policy")
        != "not implemented or benchmarked in round one; profile inline winner first"
    ):
        raise ValueError("frozen W3 LUT benchmark config drifted")
    if config.get("warmup", 0) < 1 or config.get("repeats", 0) < 1 or config.get("rounds", 0) < 3:
        raise ValueError("invalid timing repetition contract")


def tensor_file_index(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return {key: root / value for key, value in data["weight_map"].items()}
    result = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate snapshot key: {key}")
                result[key] = path
    return result


def load_indexed_tensor(index: dict[str, Path], key: str) -> torch.Tensor:
    path = index.get(key)
    if path is None:
        raise KeyError(key)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def capture(functions: dict[str, object]) -> tuple[dict[str, object], list[torch.cuda.CUDAGraph]]:
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
        raise RuntimeError("formal W3 LUT benchmark requires NVIDIA CUDA")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sources = source_hashes()
    contracts = contract_hashes()
    if environment.get("status") != "ready_for_gpu_trial" or environment.get("source_sha256") != sources:
        raise ValueError("environment/source preflight drifted")
    if environment.get("w3_contract_sha256") != contracts:
        raise ValueError("environment/W3 trial contract hashes drifted")
    if environment.get("torch", {}).get("version") != torch.__version__:
        raise ValueError("PyTorch version drifted")
    if environment.get("torch", {}).get("cuda_runtime") != torch.version.cuda:
        raise ValueError("CUDA runtime drifted")
    if environment["torch"]["devices"][0]["name"] != torch.cuda.get_device_name(0):
        raise ValueError("GPU drifted")
    cuda_preflight = json.loads(args.cuda_preflight.read_text(encoding="utf-8"))
    if (
        cuda_preflight.get("status") != "passed"
        or cuda_preflight.get("environment_sha256") != sha256_file(args.environment)
        or cuda_preflight.get("source_sha256") != sources
        or cuda_preflight.get("w3_contract_sha256") != contracts
        or cuda_preflight.get("gpu") != torch.cuda.get_device_name(0)
        or len(cuda_preflight.get("checks", [])) != 6
        or not all(item.get("passed") for item in cuda_preflight.get("checks", []))
    ):
        raise ValueError("CUDA correctness preflight missing or drifted")

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for row_tile in ROW_TILES:
        load_w3_lut_extension(row_tile)
    load_qbb_extension("v5_p1024")

    layer = int(config["layer"])
    w3_tensors, w3_entry = load_w3_lut_layer(
        args.w3_layout_root,
        layer,
        expected_manifest_sha256=args.w3_layout_manifest_sha256,
    )
    qbb_tensors, qbb_entry = load_accepted_layer(args.qbb_artifact_root, layer)
    snapshot_index = tensor_file_index(args.snapshot_root)
    decoded_path = args.gptq_root / "decoded_bf16" / f"layer-{layer:03d}.safetensors"
    report = {
        "status": "running",
        "stage": "w3_lut_inline_4x3",
        "config_sha256": sha256_file(args.config),
        "environment_sha256": sha256_file(args.environment),
        "cuda_preflight_sha256": sha256_file(args.cuda_preflight),
        "source_sha256": sources,
        "w3_contract_sha256": contracts,
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "w3_layout_layer": w3_entry,
        "qbb_layer": qbb_entry,
        "gpu": torch.cuda.get_device_name(0),
        "metric": "same-run CUDA Graph total per call",
        "timing_scope": config["timing_contract"],
        "prepare_in_round_one": False,
        "cells": [],
    }
    atomic_json(args.output, report)
    try:
        with safe_open(decoded_path, framework="pt", device="cpu") as decoded_handle:
            for shape_class, module in config["shape_classes"].items():
                layout = {field: w3_tensors[f"{module}.{field}"].cuda() for field in FIELDS}
                decoded = decoded_handle.get_tensor(f"{module}.weight").cuda()
                original = load_indexed_tensor(
                    snapshot_index, f"model.layers.{layer}.{module}.weight"
                ).cuda()
                if decoded.dtype != torch.bfloat16 or original.dtype != torch.bfloat16 or decoded.shape != original.shape:
                    raise ValueError(f"dense baseline dtype/shape drifted: {module}")
                out_features, in_features = decoded.shape
                if [out_features, in_features] != config["expected_shapes"][shape_class]:
                    raise ValueError(f"frozen Qwen3-8B shape drifted: {module}")
                qbb_payload = {field: qbb_tensors[f"{module}.{field}"] for field in QBB_FIELDS}
                qbb_layout = {key: value.cuda() for key, value in convert_qbb(qbb_payload, kernel="v5_p1024").items()}

                generator = torch.Generator(device="cuda").manual_seed(config["seed"])
                x = torch.randn(1, in_features, generator=generator, device="cuda", dtype=torch.bfloat16)
                decoded_out = torch.empty(1, out_features, device="cuda", dtype=torch.bfloat16)
                original_out = torch.empty_like(decoded_out)
                qbb_out = torch.empty_like(decoded_out)
                qbb_workspace = torch.empty(
                    qbb_workspace_shape(out_features, in_features, 1, kernel="v5_p1024"),
                    device="cuda",
                )

                def decoded_dense():
                    return torch.mm(x, decoded.t(), out=decoded_out)

                def original_dense():
                    return torch.mm(x, original.t(), out=original_out)

                def v5_anchor():
                    return qbb_m1_out(
                        x, qbb_layout, qbb_out, qbb_workspace,
                        groups_per_split=1, kernel="v5_p1024"
                    )

                reference = structural_w3_matvec(x, layout)
                for row_tile in ROW_TILES:
                    candidate_out = torch.empty_like(decoded_out)
                    candidate_workspace = torch.full(
                        workspace_shape(out_features, in_features),
                        float("nan"),
                        device="cuda",
                    )

                    def candidate():
                        return w3_lut_m1_out(
                            x, layout, candidate_out, candidate_workspace, row_tile=row_tile
                        )

                    candidate()
                    first = candidate_out.clone()
                    candidate_workspace.fill_(float("nan"))
                    candidate()
                    correctness = numerical_gate(candidate_out, reference.to(torch.bfloat16))
                    correctness.update(
                        {
                            "repeat_exact": bool(torch.equal(first, candidate_out)),
                            "workspace_finite": bool(torch.isfinite(candidate_workspace).all()),
                            "error_vs_decoded_bf16": numerical_gate(candidate_out, decoded_dense()),
                        }
                    )
                    if not correctness["passed"] or not correctness["repeat_exact"] or not correctness["workspace_finite"]:
                        raise RuntimeError(f"W3 LUT correctness failed: {shape_class}/R{row_tile}")

                    functions, graphs = capture(
                        {
                            "decoded_w3_bf16": decoded_dense,
                            "original_bf16": original_dense,
                            "v5_p1024_gps1": v5_anchor,
                            "w3_lut_inline": candidate,
                        }
                    )
                    if not numerical_gate(candidate_out, reference.to(torch.bfloat16))["passed"]:
                        raise RuntimeError(f"W3 LUT CUDA Graph correctness failed: {shape_class}/R{row_tile}")
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
                    packed_us = timings["w3_lut_inline"]["median_us"]
                    cell = {
                        "shape_class": shape_class,
                        "module": module,
                        "shape": [out_features, in_features],
                        "row_tile": row_tile,
                        "input_sha256": tensor_sha256(x),
                        "correctness": correctness,
                        "timings": timings,
                        "timing_stable": stable,
                        "speedup_vs_decoded_w3_bf16": (
                            timings["decoded_w3_bf16"]["median_us"] / packed_us if stable else None
                        ),
                        "speedup_vs_original_bf16": (
                            timings["original_bf16"]["median_us"] / packed_us if stable else None
                        ),
                        "relative_to_v5": (
                            timings["v5_p1024_gps1"]["median_us"] / packed_us if stable else None
                        ),
                    }
                    report["cells"].append(cell)
                    atomic_json(args.output, report)
                    del graphs, functions, candidate_out, candidate_workspace

                del layout, decoded, original, qbb_layout, qbb_workspace, qbb_out, reference
        if len(report["cells"]) != config["candidate_count"]:
            raise RuntimeError("4x3 candidate coverage drifted")
        report["status"] = (
            "completed_pending_review"
            if all(cell["timing_stable"] for cell in report["cells"])
            else "completed_with_unstable_cells"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
