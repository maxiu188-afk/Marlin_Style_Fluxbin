"""Fused drop-in forwards for the stock HF Qwen3 decode path.

At batch-1 decode the quantized Linears are only about 62% of the step time on
A100; the rest is dominated by how many kernels the stock modules issue, not by
bandwidth. `Qwen3RMSNorm.forward` costs eight elementwise/reduction kernels per
call and there are four norms per layer, so 36 layers spend 1152 kernels on
tensors of a few kilobytes each. Stock RoPE adds ten more per layer. Inside a
CUDA Graph each such kernel still costs roughly 2.5 us of GPU-side launch and
tail, which is where most of the non-Linear time goes.

RoPE is fused with plain tensor algebra. RMSNorm needs Inductor: contrary to a
natural guess, `torch.nn.functional.rms_norm` does not help, because
aten::rms_norm is CompositeImplicitAutograd and decomposes into the same eight
operations on CUDA.

These replacements keep the same mathematics and reduce the kernel count. They
are NOT bit-exact with the stock path: the fused kernels round at different
points. Enable them for every arm of a comparison or for none, otherwise the
arms stop being comparable. Deviation from the stock path is measured in
tests/test_fused_modules.py rather than assumed to be zero.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as functional


def rms_norm_reference(hidden_states, weight, eps):
    """Exactly the stock Qwen3RMSNorm arithmetic, as a free function.

    Kept identical on purpose: the fusion below compiles this, so the only
    change against stock is how many kernels the same operations are emitted
    as, not the operations themselves or their order.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


_COMPILED_RMS_NORM = None


def _compiled_rms_norm():
    """Compile `rms_norm_reference` once per process.

    `torch.nn.functional.rms_norm` is NOT a fused kernel: aten::rms_norm is
    CompositeImplicitAutograd with no backend registration, so it decomposes
    into the same eight operations on CUDA as on CPU, and torch 2.8 has no
    `_fused_rms_norm` at all. Inductor is therefore the mechanism that actually
    reduces the kernel count here.
    """
    global _COMPILED_RMS_NORM
    if _COMPILED_RMS_NORM is None:
        _COMPILED_RMS_NORM = torch.compile(rms_norm_reference, dynamic=False)
    return _COMPILED_RMS_NORM


def fused_rms_norm_forward(self, hidden_states):
    return _compiled_rms_norm()(hidden_states, self.weight, self.variance_epsilon)


def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Rotate q and k as one tensor: five kernels instead of ten.

    cos/sin broadcast over the head axis, so concatenating q and k along it lets
    one rotate-half and one multiply-add serve both. `narrow` returns views, so
    the split back is free.
    """
    if q.ndim != k.ndim or q.dtype != k.dtype or q.shape[-1] != k.shape[-1]:
        raise ValueError("fused RoPE requires q/k with matching rank, dtype and head_dim")
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_heads = q.shape[unsqueeze_dim]
    joined = torch.cat((q, k), dim=unsqueeze_dim)
    half = joined.shape[-1] // 2
    rotated = torch.cat((-joined[..., half:], joined[..., :half]), dim=-1)
    embedded = torch.addcmul(joined * cos, rotated, sin)
    return (
        embedded.narrow(unsqueeze_dim, 0, q_heads),
        embedded.narrow(unsqueeze_dim, q_heads, k.shape[unsqueeze_dim]),
    )


@contextmanager
def fused_qwen3_modules(*, rms_norm=True, rope=True):
    """Patch the Qwen3 classes/functions in place and restore them on exit.

    Patching happens at class/module scope, so every model built from
    `transformers` shares it. That is deliberate: a comparison whose arms do not
    all use the same non-Linear path is not a comparison.
    """
    from transformers.models.qwen3 import modeling_qwen3

    applied = {"rms_norm": bool(rms_norm), "rope": bool(rope)}
    originals = {}
    try:
        if rms_norm:
            originals["rms_norm"] = modeling_qwen3.Qwen3RMSNorm.forward
            modeling_qwen3.Qwen3RMSNorm.forward = fused_rms_norm_forward
        if rope:
            originals["rope"] = modeling_qwen3.apply_rotary_pos_emb
            modeling_qwen3.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
        yield applied
    finally:
        if "rms_norm" in originals:
            modeling_qwen3.Qwen3RMSNorm.forward = originals["rms_norm"]
        if "rope" in originals:
            modeling_qwen3.apply_rotary_pos_emb = originals["rope"]
