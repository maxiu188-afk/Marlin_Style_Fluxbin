#!/usr/bin/env python3
"""Compile and correctness-check the inline W3 LUT candidates on target CUDA.

This bounded preflight uses only synthetic W3 data.  It does not convert the
Qwen3 checkpoint and does not run any formal performance measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fluxbin_style.acceleration_checks import numerical_gate
from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256
from fluxbin_style.gptq_deployment import (
    convert_gptq_w3_to_planar,
    pack_gptq_w3_codes,
    structural_w3_matvec,
)
from fluxbin_style.w3_lut_deployment import (
    ROW_TILES,
    load_w3_lut_extension,
    w3_lut_m1_out,
    workspace_shape,
)


ROOT = Path(__file__).resolve().parents[1]
W3_CONTRACT_FILES = (
    "configs/acceleration/w3_lut_candidates_v1.json",
    "scripts/record_acceleration_environment.py",
    "scripts/prepare_qwen3_8b_w3_lut_artifacts.py",
    "scripts/run_w3_lut_server_preflight.py",
    "scripts/run_w3_lut_benchmark.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if path.suffix in {".py", ".cu", ".cuh"}
    }


def contract_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in W3_CONTRACT_FILES}


def synthetic_raw() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260916)
    in_features, out_features = 256, 288
    groups = in_features // 128
    codes = torch.randint(
        0, 8, (in_features, out_features), generator=generator, dtype=torch.uint8
    )
    activation_order = torch.randperm(in_features, generator=generator)
    g_idx = torch.empty(in_features, dtype=torch.int32)
    g_idx[activation_order] = torch.arange(groups, dtype=torch.int32).repeat_interleave(128)
    scales = (torch.rand(groups, out_features, generator=generator) * 0.125 + 0.001).half()
    raw_zero_v1 = torch.full((groups, out_features), 3, dtype=torch.uint8)
    return {
        "qweight": pack_gptq_w3_codes(codes),
        "qzeros": pack_gptq_w3_codes(raw_zero_v1, packed_axis=1),
        "scales": scales,
        "g_idx": g_idx,
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("W3 LUT server preflight requires NVIDIA CUDA")
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sources = source_hashes()
    contracts = contract_hashes()
    if environment.get("status") != "ready_for_gpu_trial":
        raise ValueError("environment is not GPU-ready")
    if environment.get("source_sha256") != sources:
        raise ValueError("environment/source hashes drifted")
    if environment.get("w3_contract_sha256") != contracts:
        raise ValueError("environment/W3 trial contract hashes drifted")
    if environment.get("torch", {}).get("version") != torch.__version__:
        raise ValueError("PyTorch version drifted")
    if environment.get("torch", {}).get("cuda_runtime") != torch.version.cuda:
        raise ValueError("CUDA runtime drifted")
    device_record = environment.get("torch", {}).get("devices", [{}])[0]
    if device_record.get("name") != torch.cuda.get_device_name(0):
        raise ValueError("GPU drifted")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for row_tile in ROW_TILES:
        load_w3_lut_extension(row_tile)

    cpu_layout = convert_gptq_w3_to_planar(synthetic_raw(), qzero_format=1)
    layout = {name: value.cuda() for name, value in cpu_layout.items()}
    out_features, in_features = 288, 256
    checks = []
    for dtype in (torch.float16, torch.bfloat16):
        generator = torch.Generator(device="cuda").manual_seed(20260916)
        x = torch.randn(1, in_features, generator=generator, device="cuda", dtype=dtype)
        reference = structural_w3_matvec(x, layout).to(dtype)
        for row_tile in ROW_TILES:
            out = torch.empty(1, out_features, device="cuda", dtype=dtype)
            workspace = torch.full(
                workspace_shape(out_features, in_features), float("nan"), device="cuda"
            )

            def run():
                return w3_lut_m1_out(x, layout, out, workspace, row_tile=row_tile)

            run()
            first = out.clone()
            eager_gate = numerical_gate(out, reference)
            workspace_finite = bool(torch.isfinite(workspace).all())
            workspace.fill_(float("nan"))
            run()
            repeat_exact = bool(torch.equal(first, out))
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            graph_gate = numerical_gate(out, reference)
            passed = eager_gate["passed"] and graph_gate["passed"] and workspace_finite and repeat_exact
            checks.append(
                {
                    "dtype": str(dtype),
                    "row_tile": row_tile,
                    "input_sha256": tensor_sha256(x),
                    "eager": eager_gate,
                    "cuda_graph": graph_gate,
                    "workspace_finite": workspace_finite,
                    "repeat_exact": repeat_exact,
                    "passed": passed,
                }
            )
            if not passed:
                raise RuntimeError(f"W3 LUT preflight failed: {dtype}/R{row_tile}")

    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "synthetic compile, eager correctness, repeatability and CUDA Graph replay; no timing",
        "environment_sha256": sha256_file(args.environment),
        "source_sha256": sources,
        "w3_contract_sha256": contracts,
        "gpu": torch.cuda.get_device_name(0),
        "shape": [out_features, in_features],
        "checks": checks,
    }
    atomic_json(args.output, report)
    print("FLUXBIN_W3_LUT_CUDA_PREFLIGHT=passed")


if __name__ == "__main__":
    main()
