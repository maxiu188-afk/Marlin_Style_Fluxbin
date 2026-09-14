"""Versioned, lossless g128/s8 layout and engine-independent M=1 interface.

No CUDA compilation, model downloads or serving-framework imports at import time.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .evaluation import materialize_hybrid_s8_weight, tensor_sha256

FORMAT = 'fluxbin-hybrid-g128-s8-m1-v1'
KERNELS = ('v1', 'v2_r1', 'v2_r2', 'v2')
FIELDS = ('global_sign_codes', 'global_row_scales', 'global_column_scales',
          'refinement_indices', 'refinement_sign_codes',
          'refinement_row_scales', 'refinement_column_scales')
LAYOUT_FIELDS = ('codes', 'rows', 'columns', 'indices', 'sparse_codes',
                 'sparse_rows', 'sparse_columns', 'lookup')


def validate_artifact(p: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    if set(p) != set(FIELDS):
        raise ValueError('expected exactly seven hybrid payload fields')
    codes = p['global_sign_codes']
    if codes.ndim != 2 or codes.dtype != torch.uint8:
        raise ValueError('global codes must be uint8 [O,K/4]')
    o, width = codes.shape
    if o < 1 or o > 65536 or width < 32 or width > 32768 or width % 32:
        raise ValueError('v1 requires O=1..65536 and K=128..131072 divisible by 128')
    g = width // 32
    shapes = ((o, width), (2, o, g), (2, g, 128), (g, 8),
              (o, g * 2), (2, o, g), (2, g, 8))
    for name, shape in zip(FIELDS, shapes):
        t = p[name]
        if tuple(t.shape) != shape or t.device != codes.device:
            raise ValueError(f'shape/device mismatch: {name}')
        if 'scales' in name and (t.dtype != torch.float32 or not torch.isfinite(t).all()):
            raise ValueError(f'finite FP32 scales required: {name}')
    if p['refinement_sign_codes'].dtype != torch.uint8:
        raise ValueError('refinement codes must be uint8')
    ix = p['refinement_indices']
    if ix.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError('integer refinement indices required')
    if (ix < 0).any() or (ix >= 128).any() or (ix[:, 1:] <= ix[:, :-1]).any():
        raise ValueError('indices must be sorted, unique and inside each group')
    return o, g


def convert_artifact(p: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Permute only; preserve sign bits, FP32 scales and selected columns."""
    o, g = validate_artifact(p)
    ix = p['refinement_indices'].to(torch.int16)
    lookup = torch.full((g, 128), -1, dtype=torch.int16, device=ix.device)
    lookup.scatter_(1, ix.long(), torch.arange(8, device=ix.device, dtype=torch.int16).expand(g, 8))
    values = (
        p['global_sign_codes'].reshape(o, g, 32).permute(1, 0, 2),
        p['global_row_scales'].permute(2, 1, 0),
        p['global_column_scales'].permute(1, 2, 0), ix,
        p['refinement_sign_codes'].reshape(o, g, 2).permute(1, 0, 2),
        p['refinement_row_scales'].permute(2, 1, 0),
        p['refinement_column_scales'].permute(1, 2, 0), lookup,
    )
    return {k: v.contiguous().clone() for k, v in zip(LAYOUT_FIELDS, values)}


def restore_artifact(p: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    g, o, _ = p['codes'].shape
    values = (p['codes'].permute(1, 0, 2).reshape(o, g * 32),
              p['rows'].permute(2, 1, 0), p['columns'].permute(2, 0, 1),
              p['indices'], p['sparse_codes'].permute(1, 0, 2).reshape(o, g * 2),
              p['sparse_rows'].permute(2, 1, 0), p['sparse_columns'].permute(2, 0, 1))
    return {k: v.contiguous() for k, v in zip(FIELDS, values)}


def decode_layout(p: Mapping[str, torch.Tensor], dtype=torch.bfloat16) -> torch.Tensor:
    return materialize_hybrid_s8_weight(**restore_artifact(p), group_size=128,
                                       columns_per_group=8, device=p['codes'].device,
                                       output_dtype=dtype)


def conversion_record(source, layout):
    return {'format': FORMAT, 'source': {k: tensor_sha256(v) for k, v in source.items()},
            'layout': {k: tensor_sha256(v) for k, v in layout.items()},
            'layout_bytes': sum(v.numel() * v.element_size() for v in layout.values())}


def load_extension(kernel="v1"):
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    return _load_extension(kernel)


@lru_cache(maxsize=4)
def _load_extension(kernel):
    if not torch.cuda.is_available():
        raise RuntimeError('M=1 kernel requires NVIDIA CUDA; no CPU/MPS substitution')
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / 'csrc'
    filename = 'm1.cu' if kernel == 'v1' else 'm1_v2.cu'
    flags = ['-O3', '--fmad=false', '-lineinfo']
    if kernel != 'v1':
        rows = {'v2_r1': 1, 'v2_r2': 2, 'v2': 4}[kernel]
        flags += [f'-DROWS_PER_WARP={rows}', '--ptxas-options=-v']
    return load(name=f'fluxbin_m1_{kernel}', sources=[str(root / filename)],
                extra_cuda_cflags=flags, verbose=True)


def workspace_shape(out_features: int, in_features: int, groups_per_split=8):
    if (not isinstance(groups_per_split,int) or not 1<=groups_per_split<=1024
            or not 1<=out_features<=65536 or not 128<=in_features<=131072 or in_features % 128):
        raise ValueError('invalid group/split dimensions')
    return ((in_features // 128 + groups_per_split - 1) // groups_per_split, out_features)


def m1_out(x, layout, out, workspace, *, groups_per_split=8, kernel="v1"):
    """Engine-neutral inference ABI. Caller owns output and per-call workspace.

    CUDA current stream; BF16/FP16 input/output; FP32 accumulation. No allocations
    after extension warmup. Only contiguous [1,K] supported. No hidden fallback.
    Workspaces must not be shared by overlapping calls on different streams.
    """
    load_extension(kernel).m1_out(x, layout['codes'], layout['rows'], layout['columns'],
                            layout['sparse_codes'], layout['sparse_rows'],
                            layout['sparse_columns'], layout['lookup'], out,
                            workspace, groups_per_split)
    return out


class PackedHybridLinear(nn.Module):
    """HF/block adapter; serving engines may call m1_out directly.

    Default rejects non-M=1 calls. Explicit dense fallback supports prefill for
    later integration checks, reconstructing on demand (not a prefill speed path).
    Keep this module's FP32 buffers intact: move with .to(device), never .half().
    """
    def __init__(self, payload, *, bias=None, fallback='error', groups_per_split=8, kernel='v1'):
        super().__init__()
        if fallback not in ('error', 'dense'):
            raise ValueError('fallback must be error or dense')
        if kernel not in KERNELS:raise ValueError('unknown kernel')
        self.kernel = kernel
        self.route_counts = {'packed_m1': 0, 'dense_fallback': 0}
        self.out_features, g = validate_artifact(payload)
        self.in_features = g * 128
        self.fallback, self.groups_per_split = fallback, groups_per_split
        workspace_shape(self.out_features, self.in_features, groups_per_split)
        for name, value in convert_artifact(payload).items():
            self.register_buffer(name, value)
        self.register_buffer('bias', None if bias is None else bias.detach().clone())
        self.register_buffer('_workspace', None, persistent=False)
        self.last_route = 'not_called'

    def layout(self):
        return {name: getattr(self, name) for name in LAYOUT_FIELDS}

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError('packed inference requires no_grad/inference_mode')
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError('expected [...,K] input')
        if any(getattr(self, n).dtype != torch.float32 for n in
               ('rows', 'columns', 'sparse_rows', 'sparse_columns')):
            raise ValueError('FP32 scale buffers were cast; move devices without dtype conversion')
        m = x.numel() // self.in_features
        if m != 1 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            if self.fallback != 'dense':
                raise ValueError('unsupported input; enable explicit dense fallback if needed')
            self.last_route = 'dense_fallback'
            self.route_counts['dense_fallback'] += 1
            return F.linear(x, decode_layout(self.layout(), x.dtype), self.bias)
        self.last_route = 'packed_m1'
        self.route_counts['packed_m1'] += 1
        shape = workspace_shape(self.out_features, self.in_features, self.groups_per_split)
        # Adapter owns one workspace: same-stream sequential model execution only.
        if self._workspace is None or self._workspace.device != x.device:
            self._workspace = torch.empty(shape, device=x.device, dtype=torch.float32)
        y = torch.empty((1, self.out_features), device=x.device, dtype=x.dtype)
        m1_out(x.reshape(1, self.in_features).contiguous(), self.layout(), y,
               self._workspace, groups_per_split=self.groups_per_split, kernel=self.kernel)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


def replace_block_linears(block, layer_payload, *, fallback='error', groups_per_split=8, kernel='v1'):
    """Replace exactly seven Qwen3 Linears; usable for one block or all 36 blocks."""
    from .qwen3 import QWEN3_LINEAR_MODULES
    expected = {f'{module}.{field}' for module in QWEN3_LINEAR_MODULES for field in FIELDS}
    if set(layer_payload) != expected:
        raise ValueError('layer must contain exactly seven complete hybrid payloads')
    replacements = []
    for name in QWEN3_LINEAR_MODULES:
        old = block.get_submodule(name)
        p = {field: layer_payload[f'{name}.{field}'] for field in FIELDS}
        o, g = validate_artifact(p)
        if not isinstance(old, nn.Linear) or (old.out_features, old.in_features) != (o, g * 128):
            raise ValueError(f'Linear shape/type mismatch: {name}')
        replacement = PackedHybridLinear(p, bias=old.bias, fallback=fallback,
                                         groups_per_split=groups_per_split,kernel=kernel).to(old.weight.device)
        replacements.append((name, replacement))
    # Validate/build every replacement before mutating the block.
    for name, replacement in replacements:
        parent_name, attr = name.rsplit('.', 1)
        setattr(block.get_submodule(parent_name), attr, replacement)
    return [name for name, _ in replacements]
