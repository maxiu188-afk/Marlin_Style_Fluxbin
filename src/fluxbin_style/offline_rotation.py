"""Offline-only QuaRot-style reparameterization for Qwen3.

Ported from the validated Llama implementation in the QuaRot/SpinQuant
reproduction workspace (``repro/offline_llama_rotation.py``).  Only the two
rotations that fold into ordinary parameters are applied:

``R1``  the residual/hidden rotation, absorbed into the embedding, every
        Linear that reads the residual stream, every Linear that writes it,
        and ``lm_head``;
``R2``  the per-head value/output rotation, absorbed into ``v_proj`` outputs
        and ``o_proj`` inputs.

The online MLP (``R4``) and QK (``R3``) Hadamards are deliberately excluded.
Their inputs are elementwise activations that cannot be folded into any weight
matrix, so they would need a runtime kernel.  ``down_proj`` therefore keeps its
original input basis, and this module adds no kernel, no activation
quantization, and no serving code.

The transform is exact in real arithmetic.  With the RMSNorm scale fused into
the following Linear, a unit RMSNorm commutes with an orthogonal rotation
(``rms(xR) == rms(x)``), so a rotated model reproduces the original logits up
to floating-point error.  Qwen3's per-head ``q_norm``/``k_norm`` sit after
``q_proj``/``k_proj`` and are left untouched, because ``R1`` only changes those
projections' input basis.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from fluxbin_style.evaluation import tensor_sha256


QWEN3_ROTATION_FORMAT = "offline-qwen3-r1-r2-v1"


def normalized_hadamard_matrix(
    size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a symmetric orthogonal Walsh-Hadamard matrix of ``size``."""

    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(size)


def _fuse_norm_scale(norm: nn.Module, linears: tuple[nn.Linear, ...]) -> None:
    if not hasattr(norm, "weight") or norm.weight is None:
        raise ValueError("offline rotation requires affine RMSNorm weights")
    scale = norm.weight.detach().clone()
    for linear in linears:
        fused = linear.weight.float() * scale.float()
        linear.weight.copy_(fused.to(linear.weight.dtype))
    norm.weight.fill_(1)


def _copy_matmul_(
    target: torch.Tensor, *factors: torch.Tensor, device: torch.device | None = None
) -> None:
    """Accumulate ``factors`` in FP32 and write the product back into ``target``.

    ``device`` borrows an accelerator for the product only; every parameter
    stays wherever the caller put it, so an 8B model can be rotated from host
    memory without materializing the whole model on the GPU.
    """

    work = factors[0].to(device=device, dtype=torch.float32)
    for factor in factors[1:]:
        work = work @ factor.to(device=device, dtype=torch.float32)
    if work.shape != target.shape and work.numel() == target.numel():
        work = work.reshape(target.shape)
    target.copy_(work.to(device=target.device, dtype=target.dtype))


def _untie_output_head_if_needed(model: nn.Module) -> bool:
    """``R1`` rotates the input and output embeddings differently."""

    if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
        return False
    replacement = nn.Linear(
        model.lm_head.in_features,
        model.lm_head.out_features,
        bias=model.lm_head.bias is not None,
        device=model.lm_head.weight.device,
        dtype=model.lm_head.weight.dtype,
    )
    replacement.weight.copy_(model.lm_head.weight)
    if replacement.bias is not None:
        replacement.bias.copy_(model.lm_head.bias)
    model.lm_head = replacement
    model.config.tie_word_embeddings = False
    return True


def assert_standard_qwen3_layout(model: nn.Module) -> None:
    """Reject any module layout this offline transform cannot fold into."""

    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError("offline rotation expects a qwen3 model")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("model does not expose the standard Qwen3 decoder layout")
    if not isinstance(model.lm_head, nn.Linear):
        raise ValueError("lm_head must be a Linear")
    for layer in model.model.layers:
        for linear in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            if not isinstance(linear, nn.Linear):
                raise ValueError("unexpected non-Linear projection")
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if not hasattr(norm, "weight") or norm.weight is None:
                raise ValueError("offline rotation requires affine RMSNorm weights")
        # q_norm/k_norm act on head_dim after the projection and stay untouched,
        # but their presence is part of the layout this port was checked against.
        for name in ("q_norm", "k_norm"):
            if not hasattr(layer.self_attn, name):
                raise ValueError(f"Qwen3 attention is missing {name}")


@torch.no_grad()
def apply_offline_qwen3_rotation(
    model: nn.Module,
    *,
    rotate_values: bool = True,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Fold ``R1`` (and optionally ``R2``) into a Qwen3 model in place.

    ``compute_device`` only borrows an accelerator for the products; the model
    itself is never moved, so a host-resident 8B checkpoint can be rotated
    without holding it all in device memory.
    """

    assert_standard_qwen3_layout(model)
    config = model.config
    hidden_size = int(config.hidden_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // num_q_heads))
    attention = model.model.layers[0].self_attn
    if attention.q_proj.out_features != num_q_heads * head_dim:
        raise ValueError("q_proj does not match num_attention_heads x head_dim")
    if attention.v_proj.out_features != num_kv_heads * head_dim:
        raise ValueError("v_proj does not match num_key_value_heads x head_dim")
    if head_dim & (head_dim - 1):
        raise ValueError("offline V/O rotation requires a power-of-two head_dim")

    device = (
        torch.device(compute_device)
        if compute_device is not None
        else next(model.parameters()).device
    )
    compute_dtype = torch.float32
    residual = normalized_hadamard_matrix(hidden_size, compute_dtype, device)
    head = normalized_hadamard_matrix(head_dim, compute_dtype, device)
    q_head_block = torch.block_diag(*([head] * num_q_heads))
    kv_head_block = torch.block_diag(*([head] * num_kv_heads))

    output_head_was_untied = _untie_output_head_if_needed(model)

    # The scale must leave the norms before the rotation, because only a unit
    # RMSNorm commutes with an orthogonal transform.
    for layer in model.model.layers:
        _fuse_norm_scale(
            layer.input_layernorm,
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
        )
        _fuse_norm_scale(
            layer.post_attention_layernorm,
            (layer.mlp.up_proj, layer.mlp.gate_proj),
        )
    _fuse_norm_scale(model.model.norm, (model.lm_head,))

    _copy_matmul_(
        model.model.embed_tokens.weight,
        model.model.embed_tokens.weight,
        residual,
        device=device,
    )
    _copy_matmul_(model.lm_head.weight, model.lm_head.weight, residual, device=device)
    for layer in model.model.layers:
        attention, mlp = layer.self_attn, layer.mlp
        for linear in (attention.q_proj, attention.k_proj, mlp.up_proj, mlp.gate_proj):
            _copy_matmul_(linear.weight, linear.weight, residual, device=device)

        if rotate_values:
            # The same per-head block is applied to every value head, so the
            # transform commutes with the GQA repeat that feeds o_proj.
            _copy_matmul_(
                attention.v_proj.weight,
                kv_head_block,
                attention.v_proj.weight,
                residual,
                device=device,
            )
            _copy_matmul_(
                attention.o_proj.weight,
                residual.T,
                attention.o_proj.weight,
                q_head_block,
                device=device,
            )
            if attention.v_proj.bias is not None:
                _copy_matmul_(
                    attention.v_proj.bias,
                    kv_head_block,
                    attention.v_proj.bias.unsqueeze(1),
                    device=device,
                )
        else:
            _copy_matmul_(
                attention.v_proj.weight, attention.v_proj.weight, residual, device=device
            )
            _copy_matmul_(
                attention.o_proj.weight, residual.T, attention.o_proj.weight, device=device
            )
        if attention.o_proj.bias is not None:
            _copy_matmul_(
                attention.o_proj.bias,
                residual.T,
                attention.o_proj.bias.unsqueeze(1),
                device=device,
            )

        _copy_matmul_(mlp.down_proj.weight, residual.T, mlp.down_proj.weight, device=device)
        if mlp.down_proj.bias is not None:
            _copy_matmul_(
                mlp.down_proj.bias,
                residual.T,
                mlp.down_proj.bias.unsqueeze(1),
                device=device,
            )

    return {
        "format": QWEN3_ROTATION_FORMAT,
        "residual_rotation": "walsh",
        "residual_size": hidden_size,
        "residual_sha256": tensor_sha256(residual.cpu()),
        "head_rotation_size": head_dim,
        "head_rotation_sha256": tensor_sha256(head.cpu()),
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "rotate_values": rotate_values,
        "output_head_was_untied": output_head_was_untied,
        "online_mlp_hadamard": False,
        "online_qk_hadamard": False,
        "down_proj_input_rotated": False,
        "activation_quantization": False,
        "custom_kernel": False,
    }
