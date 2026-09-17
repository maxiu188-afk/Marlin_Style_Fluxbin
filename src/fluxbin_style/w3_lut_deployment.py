"""CUDA extension wrapper for the bounded inline GPTQ W3 LUT experiment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .gptq_deployment import FIELDS, restore_planar_w3, validate_planar_w3


ROW_TILES = (256, 512, 1024)
EXPERIMENTAL_ROW_TILES = ROW_TILES + (2048,)
ARITHMETIC_MODES = ("structural", "decoded_bf16")
QWEN3_ROW_TILE_BY_SHAPE = {
    (4096, 4096): 256,
    (1024, 4096): 512,
    (12288, 4096): 1024,
    (4096, 12288): 1024,
}
QWEN3_W3_ROUTE_POLICIES = {
    "structural": {
        shape: {"row_tile": row_tile, "groups_per_split": 1, "arithmetic": "structural"}
        for shape, row_tile in QWEN3_ROW_TILE_BY_SHAPE.items()
    },
    "fast_corrected": {
        shape: {"row_tile": row_tile, "groups_per_split": 1, "arithmetic": "decoded_bf16"}
        for shape, row_tile in QWEN3_ROW_TILE_BY_SHAPE.items()
    },
    "observed_exact": {
        shape: {
            "row_tile": row_tile,
            "groups_per_split": {
                (4096, 4096): 1,
                (1024, 4096): 1,
                (12288, 4096): 4,
                (4096, 12288): 2,
            }[shape],
            "arithmetic": "decoded_bf16",
        }
        for shape, row_tile in QWEN3_ROW_TILE_BY_SHAPE.items()
    },
}


@lru_cache(maxsize=2 * len(EXPERIMENTAL_ROW_TILES))
def load_w3_lut_extension(row_tile: int, uniform_index_probe: bool = False):
    """Build the W3 LUT extension for one row tile.

    `uniform_index_probe` selects the timing-only build described in w3_lut.cu:
    it removes shared-memory bank conflicts by broadcasting the LUT index and
    therefore returns wrong values. It gets its own extension name so it can
    never share a build cache entry, or a loaded module, with the real kernel.
    """
    if row_tile not in EXPERIMENTAL_ROW_TILES:
        raise ValueError(f"row_tile must be one of {EXPERIMENTAL_ROW_TILES}")
    if not torch.cuda.is_available():
        raise RuntimeError("W3 LUT requires NVIDIA CUDA; no CPU/MPS substitution")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent / "csrc"
    return load(
        name=f"fluxbin_w3_lut_r{row_tile}" + ("_uniformprobe" if uniform_index_probe else ""),
        sources=[str(root / "w3_lut.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "-lineinfo",
            "--ptxas-options=-v",
            f"-DW3_LUT_ROWS={row_tile}",
            f"-DW3_LUT_UNIFORM_INDEX_PROBE={int(bool(uniform_index_probe))}",
        ],
        verbose=True,
    )


def workspace_shape(
    out_features: int,
    in_features: int,
    groups_per_split: int = 1,
) -> tuple[int, int]:
    if not 1 <= out_features <= 65536 or not 128 <= in_features <= 32767 or in_features % 128:
        raise ValueError("invalid W3 LUT dimensions")
    groups = in_features // 128
    if not isinstance(groups_per_split, int) or not 1 <= groups_per_split <= groups:
        raise ValueError("groups_per_split must be an integer in [1,K/128]")
    return (groups + groups_per_split - 1) // groups_per_split, out_features


def inline_main(
    x: torch.Tensor,
    layout: dict[str, torch.Tensor],
    workspace: torch.Tensor,
    *,
    row_tile: int,
    arithmetic: str = "structural",
) -> None:
    # Value-level checks are performed once by the hash-bound artifact loader.
    # Keep this hot path free of CUDA reductions and temporary allocations so
    # graph capture contains only the candidate main and finish kernels.
    out_features, in_features = validate_planar_w3(layout, check_values=False)
    if x.shape != (1, in_features):
        raise ValueError("x must have shape [1,K]")
    if arithmetic not in ARITHMETIC_MODES:
        raise ValueError(f"arithmetic must be one of {ARITHMETIC_MODES}")
    if workspace.shape != workspace_shape(out_features, in_features) or workspace.dtype != torch.float32:
        raise ValueError("workspace must be FP32 [K/128,O]")
    if workspace.device != x.device or any(value.device != x.device for value in layout.values()):
        raise ValueError("x, layout and workspace must share a device")
    extension = load_w3_lut_extension(row_tile)
    function = (
        extension.inline_main
        if arithmetic == "structural"
        else extension.inline_main_bf16_weight
    )
    function(x, layout["planes"], layout["scales"], layout["perm"], workspace)


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
    groups_per_split: int = 1,
    arithmetic: str = "structural",
) -> torch.Tensor:
    """Allocation-free CUDA-current-stream M=1 experimental interface."""
    if out.dtype != x.dtype:
        raise ValueError("out dtype must match x dtype")
    if arithmetic not in ARITHMETIC_MODES:
        raise ValueError(f"arithmetic must be one of {ARITHMETIC_MODES}")
    out_features, in_features = validate_planar_w3(layout, check_values=False)
    if x.shape != (1, in_features) or out.shape != (1, out_features):
        raise ValueError("x/out shape mismatch")
    expected_workspace = workspace_shape(out_features, in_features, groups_per_split)
    if workspace.shape != expected_workspace or workspace.dtype != torch.float32:
        raise ValueError("workspace must be FP32 [ceil(G/groups_per_split),O]")
    if workspace.device != x.device or out.device != x.device:
        raise ValueError("x, out and workspace must share a device")
    load_w3_lut_extension(row_tile).m1_out(
        x,
        layout["planes"],
        layout["scales"],
        layout["perm"],
        workspace,
        out,
        groups_per_split,
        arithmetic == "decoded_bf16",
    )
    return out


def qwen3_row_tile(out_features: int, in_features: int) -> int:
    try:
        return QWEN3_ROW_TILE_BY_SHAPE[(out_features, in_features)]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3-8B W3 shape: {(out_features, in_features)}") from exc


def qwen3_w3_route(route: str, out_features: int, in_features: int) -> dict:
    if route not in QWEN3_W3_ROUTE_POLICIES:
        raise ValueError(f"unknown Qwen3-8B W3 route: {route}")
    try:
        return dict(QWEN3_W3_ROUTE_POLICIES[route][(out_features, in_features)])
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3-8B W3 shape: {(out_features, in_features)}") from exc


class PackedW3Linear(nn.Module):
    """Decode-only W3 Linear with explicit dense prefill fallback.

    The prepared binding owns one output and one FP32 partial workspace.  It is
    intentionally serial and batch-one, matching the accepted full-model CUDA
    Graph protocol.
    """

    def __init__(
        self,
        layout: dict[str, torch.Tensor],
        *,
        bias: torch.Tensor | None = None,
        fallback: str = "error",
        row_tile: int | None = None,
        groups_per_split: int = 1,
        arithmetic: str = "structural",
        route: str = "structural",
    ):
        super().__init__()
        if fallback not in {"error", "dense"}:
            raise ValueError("fallback must be error or dense")
        self.out_features, self.in_features = validate_planar_w3(layout)
        self.row_tile = (
            qwen3_row_tile(self.out_features, self.in_features)
            if row_tile is None
            else row_tile
        )
        if self.row_tile not in ROW_TILES:
            raise ValueError(f"row_tile must be one of {ROW_TILES}")
        if arithmetic not in ARITHMETIC_MODES:
            raise ValueError(f"arithmetic must be one of {ARITHMETIC_MODES}")
        workspace_shape(self.out_features, self.in_features, groups_per_split)
        self.groups_per_split = groups_per_split
        self.arithmetic = arithmetic
        self.route = route
        self.fallback = fallback
        self.route_counts = {"packed_m1": 0, "dense_fallback": 0}
        self.last_route = "not_called"
        self._decode_binding = None
        self._decode_audit = False
        for name in FIELDS:
            self.register_buffer(name, layout[name].detach().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("_workspace", None, persistent=False)
        self.register_buffer("_decode_output", None, persistent=False)

    def layout(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in FIELDS}

    def clear_prepared_decode(self) -> None:
        self._decode_binding = None
        self._decode_output = None
        self._workspace = None
        self._decode_audit = False

    def _apply(self, fn, recurse=True):
        self.clear_prepared_decode()
        return super()._apply(fn, recurse=recurse)

    def bind_prepared_decode(self, dtype: torch.dtype) -> None:
        if self._decode_binding is not None:
            raise ValueError("decode already bound")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP16/BF16 required")
        layout = self.layout()
        device = layout["planes"].device
        if device.type != "cuda":
            raise ValueError("prepared decode requires CUDA")
        validate_planar_w3(layout, check_values=False)
        if any(value.device != device or not value.is_contiguous() for value in layout.values()):
            raise ValueError("contiguous same-device layout required")
        extension = load_w3_lut_extension(self.row_tile)
        self._workspace = torch.empty(
            workspace_shape(
                self.out_features,
                self.in_features,
                self.groups_per_split,
            ),
            device=device,
            dtype=torch.float32,
        )
        self._decode_output = torch.empty(
            (1, self.out_features), device=device, dtype=dtype
        )
        output = self._decode_output
        shaped = output.view(1, 1, self.out_features)
        expected = (1, 1, self.in_features)

        def apply(x: torch.Tensor) -> torch.Tensor:
            if torch.is_grad_enabled():
                raise RuntimeError("prepared decode is inference only")
            if (
                tuple(x.shape) != expected
                or x.dtype != dtype
                or x.device != device
                or not x.is_contiguous()
            ):
                raise ValueError("prepared decode input shape/dtype/device/stride mismatch")
            if self._decode_audit:
                self.route_counts["packed_m1"] += 1
                self.last_route = "packed_m1"
            flat = x.view(1, self.in_features)
            if self.arithmetic == "structural" and self.groups_per_split == 1:
                extension.inline_main(
                    flat,
                    layout["planes"],
                    layout["scales"],
                    layout["perm"],
                    self._workspace,
                )
                extension.finish(self._workspace, output)
            else:
                extension.m1_out(
                    flat,
                    layout["planes"],
                    layout["scales"],
                    layout["perm"],
                    self._workspace,
                    output,
                    self.groups_per_split,
                    self.arithmetic == "decoded_bf16",
                )
            if self.bias is not None:
                output.add_(self.bias)
            return shaped

        self._decode_binding = apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._decode_binding is not None:
            return self._decode_binding(x)
        if torch.is_grad_enabled():
            raise RuntimeError("packed inference requires no_grad/inference_mode")
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError("expected [...,K] input")
        layout = self.layout()
        m = x.numel() // self.in_features
        if m != 1 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            if self.fallback != "dense":
                raise ValueError("unsupported input; enable explicit dense fallback if needed")
            self.last_route = "dense_fallback"
            self.route_counts["dense_fallback"] += 1
            return F.linear(x, restore_planar_w3(layout, dtype=x.dtype), self.bias)
        self.last_route = "packed_m1"
        self.route_counts["packed_m1"] += 1
        shape = workspace_shape(
            self.out_features,
            self.in_features,
            self.groups_per_split,
        )
        if (
            self._workspace is None
            or self._workspace.device != x.device
            or tuple(self._workspace.shape) != shape
        ):
            self._workspace = torch.empty(shape, device=x.device, dtype=torch.float32)
        output = torch.empty((1, self.out_features), device=x.device, dtype=x.dtype)
        w3_lut_m1_out(
            x.reshape(1, self.in_features).contiguous(),
            layout,
            output,
            self._workspace,
            row_tile=self.row_tile,
            groups_per_split=self.groups_per_split,
            arithmetic=self.arithmetic,
        )
        if self.bias is not None:
            output.add_(self.bias)
        return output.reshape(*x.shape[:-1], self.out_features)


def replace_w3_block_linears(
    block,
    layer_payload,
    *,
    fallback: str = "error",
    row_tile_by_shape: dict[tuple[int, int], int] | None = None,
    route: str = "structural",
) -> list[str]:
    """Replace exactly seven Qwen3 Linears after validating the complete layer."""
    from .qwen3 import QWEN3_LINEAR_MODULES

    expected = {f"{module}.{field}" for module in QWEN3_LINEAR_MODULES for field in FIELDS}
    if set(layer_payload) != expected:
        raise ValueError("layer must contain exactly seven complete W3 payloads")
    if route not in QWEN3_W3_ROUTE_POLICIES:
        raise ValueError(f"unknown Qwen3-8B W3 route: {route}")
    row_policy = QWEN3_ROW_TILE_BY_SHAPE if row_tile_by_shape is None else row_tile_by_shape
    replacements = []
    for name in QWEN3_LINEAR_MODULES:
        old = block.get_submodule(name)
        layout = {field: layer_payload[f"{name}.{field}"] for field in FIELDS}
        out_features, in_features = validate_planar_w3(layout)
        if not isinstance(old, nn.Linear) or (
            old.out_features,
            old.in_features,
        ) != (out_features, in_features):
            raise ValueError(f"Linear shape/type mismatch: {name}")
        try:
            row_tile = row_policy[(out_features, in_features)]
        except KeyError as exc:
            raise ValueError(
                f"row-tile policy missing shape: {(out_features, in_features)}"
            ) from exc
        if row_tile_by_shape is None:
            route_policy = qwen3_w3_route(route, out_features, in_features)
            if route_policy["row_tile"] != row_tile:
                raise ValueError("route and row-tile policies disagree")
        else:
            route_policy = {
                "groups_per_split": 1,
                "arithmetic": "structural",
            }
        replacement = PackedW3Linear(
            layout,
            bias=old.bias,
            fallback=fallback,
            row_tile=row_tile,
            groups_per_split=route_policy["groups_per_split"],
            arithmetic=route_policy["arithmetic"],
            route=route,
        ).to(old.weight.device)
        replacements.append((name, replacement))
    for name, replacement in replacements:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(block.get_submodule(parent_name), attribute, replacement)
    return [name for name, _ in replacements]
