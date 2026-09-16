"""CUDA extension wrapper for the bounded inline GPTQ W3 LUT experiment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .gptq_deployment import validate_planar_w3


ROW_TILES = (256, 512, 1024)


@lru_cache(maxsize=len(ROW_TILES))
def load_w3_lut_extension(row_tile: int):
    if row_tile not in ROW_TILES:
        raise ValueError(f"row_tile must be one of {ROW_TILES}")
    if not torch.cuda.is_available():
        raise RuntimeError("W3 LUT requires NVIDIA CUDA; no CPU/MPS substitution")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent / "csrc"
    return load(
        name=f"fluxbin_w3_lut_r{row_tile}",
        sources=[str(root / "w3_lut.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "-lineinfo",
            "--ptxas-options=-v",
            f"-DW3_LUT_ROWS={row_tile}",
        ],
        verbose=True,
    )


def workspace_shape(out_features: int, in_features: int) -> tuple[int, int]:
    if not 1 <= out_features <= 65536 or not 128 <= in_features <= 32767 or in_features % 128:
        raise ValueError("invalid W3 LUT dimensions")
    return in_features // 128, out_features


def inline_main(
    x: torch.Tensor,
    layout: dict[str, torch.Tensor],
    workspace: torch.Tensor,
    *,
    row_tile: int,
) -> None:
    # Value-level checks are performed once by the hash-bound artifact loader.
    # Keep this hot path free of CUDA reductions and temporary allocations so
    # graph capture contains only the candidate main and finish kernels.
    out_features, in_features = validate_planar_w3(layout, check_values=False)
    if x.shape != (1, in_features):
        raise ValueError("x must have shape [1,K]")
    if workspace.shape != workspace_shape(out_features, in_features) or workspace.dtype != torch.float32:
        raise ValueError("workspace must be FP32 [K/128,O]")
    if workspace.device != x.device or any(value.device != x.device for value in layout.values()):
        raise ValueError("x, layout and workspace must share a device")
    load_w3_lut_extension(row_tile).inline_main(
        x, layout["planes"], layout["scales"], layout["perm"], workspace
    )


def finish(workspace: torch.Tensor, out: torch.Tensor, *, row_tile: int) -> torch.Tensor:
    if workspace.ndim != 2 or workspace.dtype != torch.float32:
        raise ValueError("workspace must be FP32 [G,O]")
    if out.shape != (1, workspace.shape[1]) or out.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("out must be FP16/BF16 [1,O]")
    load_w3_lut_extension(row_tile).finish(workspace, out)
    return out


def w3_lut_m1_out(
    x: torch.Tensor,
    layout: dict[str, torch.Tensor],
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    row_tile: int,
) -> torch.Tensor:
    """Allocation-free CUDA-current-stream M=1 interface; inline only."""
    if out.dtype != x.dtype:
        raise ValueError("out dtype must match x dtype")
    inline_main(x, layout, workspace, row_tile=row_tile)
    return finish(workspace, out, row_tile=row_tile)
