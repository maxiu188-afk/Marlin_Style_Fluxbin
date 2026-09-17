"""Hierarchical-scale W2 projection and GPTQ error compensation.

The representation is intentionally a quality experiment, not a CUDA layout:

``W_hat = q * S128 * R32 * R16`` with ``q in {-3, -1, 1, 3}``.

Each output row has one FP16 parent scale per 128 input columns, four R32
codes, and eight R16 codes.  R32/R16 are unsigned stored codes for signed,
uniformly quantized log2 residuals.  The two quantizer steps are per Linear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


W2_LEVELS = (-3.0, -1.0, 1.0, 3.0)
GROUP_SIZE = 128
R32_WIDTH = 32
R16_WIDTH = 16


def _validate_bits(bits: int) -> None:
    if bits not in (2, 4, 8):
        raise ValueError("only 2, 4, and 8 bit unsigned fields are supported")


def pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a uint-like tensor into a flat uint8 tensor, low field first."""

    _validate_bits(bits)
    maximum = (1 << bits) - 1
    flat = values.detach().to(torch.int64).reshape(-1)
    if flat.numel() and (torch.any(flat < 0) or torch.any(flat > maximum)):
        raise ValueError(f"values do not fit in {bits} bits")
    per_byte = 8 // bits
    padding = (-flat.numel()) % per_byte
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=flat.dtype, device=flat.device)))
    fields = flat.reshape(-1, per_byte)
    packed = torch.zeros(fields.shape[0], dtype=torch.uint8, device=flat.device)
    for index in range(per_byte):
        packed |= (fields[:, index] << (index * bits)).to(torch.uint8)
    return packed


def unpack_unsigned(
    packed: torch.Tensor,
    bits: int,
    *,
    count: int,
    shape: Iterable[int] | None = None,
) -> torch.Tensor:
    """Unpack low-field-first unsigned values as uint8."""

    _validate_bits(bits)
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed must be a flat uint8 tensor")
    if count < 0:
        raise ValueError("count must be non-negative")
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    expanded = torch.empty(packed.numel() * per_byte, dtype=torch.uint8, device=packed.device)
    for index in range(per_byte):
        expanded[index::per_byte] = (packed >> (index * bits)) & mask
    result = expanded[:count]
    if shape is not None:
        target = tuple(shape)
        if result.numel() != torch.tensor(target).prod().item():
            raise ValueError("shape does not match count")
        result = result.reshape(target)
    return result


def nominal_bpw(r32_bits: int, r16_bits: int) -> float:
    """Return the semantic representation rate requested by the experiment."""

    if r32_bits not in (4, 8) or r16_bits not in (4, 8):
        raise ValueError("relative scales must use U4 or U8")
    return 2.0 + 16.0 / 128.0 + r32_bits / 32.0 + r16_bits / 16.0


def _nearest_w2(target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    normalized = target / scale.clamp_min(torch.finfo(torch.float32).tiny)
    magnitude = torch.where(normalized.abs() < 2.0, 1.0, 3.0)
    sign = torch.where(normalized < 0.0, -1.0, 1.0)
    return sign * magnitude


def _ideal_leaf_scales(target: torch.Tensor, iterations: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternating least-squares W2 scale for each consecutive 16 weights."""

    rows, columns = target.shape
    if columns % R16_WIDTH:
        raise ValueError("target width must be divisible by 16")
    leaves = target.to(torch.float32).reshape(rows, columns // R16_WIDTH, R16_WIDTH)
    q = torch.where(leaves < 0.0, -1.0, 1.0)
    scale = torch.ones((rows, leaves.shape[1], 1), dtype=torch.float32, device=target.device)
    for _ in range(iterations):
        scale = (leaves * q).sum(dim=-1, keepdim=True) / q.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        scale = scale.abs().clamp_min(torch.finfo(torch.float32).tiny)
        q = _nearest_w2(leaves, scale)
    scale = (leaves * q).sum(dim=-1, keepdim=True) / q.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    return scale.abs().clamp_min(torch.finfo(torch.float32).tiny).squeeze(-1), q.reshape(rows, columns)


def _log_components(target: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ideal b32/c16 residuals using q^2 weighted log means."""

    rows, columns = target.shape
    leaves = target.to(torch.float32).reshape(rows, columns // R16_WIDTH, R16_WIDTH)
    leaf_q = q.to(torch.float32).reshape_as(leaves)
    leaf_weights = leaf_q.square().sum(dim=-1)
    leaf_scale = (leaves * leaf_q).sum(dim=-1) / leaf_weights.clamp_min(1.0)
    leaf_scale = leaf_scale.abs().clamp_min(torch.finfo(torch.float32).tiny)
    logs = torch.log2(leaf_scale)
    parent = (logs * leaf_weights).sum(dim=-1, keepdim=True) / leaf_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    residual = logs - parent
    residual32 = residual.reshape(rows, -1, 2)
    weights32 = leaf_weights.reshape(rows, -1, 2)
    b32 = (residual32 * weights32).sum(dim=-1) / weights32.sum(dim=-1).clamp_min(1.0)
    c16 = residual - b32.repeat_interleave(2, dim=-1)
    return b32, c16


def _quantize_log_residual(value: torch.Tensor, bits: int, step: torch.Tensor | float) -> torch.Tensor:
    zero = 1 << (bits - 1)
    signed = torch.round(value / torch.as_tensor(step, dtype=torch.float32, device=value.device))
    signed = signed.clamp(-zero, zero - 1)
    return (signed + zero).to(torch.uint8)


def _decode_log_residual(code: torch.Tensor, bits: int, step: torch.Tensor | float) -> torch.Tensor:
    zero = 1 << (bits - 1)
    return (code.to(torch.float32) - zero) * torch.as_tensor(
        step, dtype=torch.float32, device=code.device
    )


def _choose_step(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    bits: int,
    multipliers: tuple[float, ...],
) -> torch.Tensor:
    """Choose a deterministic uniform step by weighted residual grid search."""

    qmax = (1 << (bits - 1)) - 1
    maximum = values.abs().amax()
    if not torch.isfinite(maximum):
        raise ValueError("non-finite log residual")
    base = (maximum / max(qmax, 1)).clamp_min(2.0**-20)
    best_step = base
    best_error: torch.Tensor | None = None
    for multiplier in multipliers:
        step = base * multiplier
        code = _quantize_log_residual(values, bits, step)
        decoded = _decode_log_residual(code, bits, step)
        error = ((values - decoded).square() * weights).sum()
        if best_error is None or bool(error < best_error):
            best_error = error
            best_step = step
    return best_step.to(torch.float32)


def choose_linear_steps(
    weight: torch.Tensor,
    *,
    r32_bits: int,
    r16_bits: int,
    multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one R32 and one R16 log-residual step for an entire Linear."""

    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError("weight must be [out, in] with in divisible by 128")
    blocks = weight.to(torch.float32).reshape(-1, GROUP_SIZE)
    _, q = _ideal_leaf_scales(blocks)
    b32, c16 = _log_components(blocks, q)
    q2 = q.reshape(-1, 8, 16).square().sum(dim=-1)
    b_weights = q2.reshape(-1, 4, 2).sum(dim=-1)
    chosen = (
        _choose_step(b32, b_weights, bits=r32_bits, multipliers=multipliers),
        _choose_step(c16, q2, bits=r16_bits, multipliers=multipliers),
    )
    # The representation stores these tiny per-Linear metadata values as FP16;
    # make the projection use exactly the values that will be serialized.
    return tuple(value.to(torch.float16).to(torch.float32) for value in chosen)  # type: ignore[return-value]


@dataclass(frozen=True)
class HierarchicalProjection:
    q: torch.Tensor
    parent: torch.Tensor
    r32_codes: torch.Tensor
    r16_codes: torch.Tensor
    effective_scale: torch.Tensor

    @property
    def reconstructed(self) -> torch.Tensor:
        return self.q * self.effective_scale


def project_hierarchical_group(
    target: torch.Tensor,
    *,
    r32_bits: int,
    r16_bits: int,
    r32_step: torch.Tensor | float,
    r16_step: torch.Tensor | float,
    alternating_rounds: int = 3,
) -> HierarchicalProjection:
    """Project a matrix of independent 128-column groups to hierarchical W2."""

    if target.ndim != 2 or target.shape[1] != GROUP_SIZE:
        raise ValueError("target must have shape [rows, 128]")
    if alternating_rounds <= 0:
        raise ValueError("alternating_rounds must be positive")
    _, q = _ideal_leaf_scales(target)
    for _ in range(alternating_rounds):
        b32, c16 = _log_components(target, q)
        b_code = _quantize_log_residual(b32, r32_bits, r32_step)
        c_code = _quantize_log_residual(c16, r16_bits, r16_step)
        relative = torch.exp2(
            _decode_log_residual(b_code, r32_bits, r32_step).repeat_interleave(32, dim=-1)
            + _decode_log_residual(c_code, r16_bits, r16_step).repeat_interleave(16, dim=-1)
        )
        z = q * relative
        parent = (target.to(torch.float32) * z).sum(dim=-1, keepdim=True) / z.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        parent = parent.abs().clamp_min(torch.finfo(torch.float16).tiny)
        parent = parent.to(torch.float16).to(torch.float32)
        effective = parent * relative
        q = _nearest_w2(target.to(torch.float32), effective)
    z = q * relative
    parent = (target.to(torch.float32) * z).sum(dim=-1, keepdim=True) / z.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    parent = parent.abs().clamp_min(torch.finfo(torch.float16).tiny)
    parent = parent.to(torch.float16).to(torch.float32)
    effective = parent * relative
    return HierarchicalProjection(
        q=q,
        parent=parent,
        r32_codes=b_code,
        r16_codes=c_code,
        effective_scale=effective,
    )


@dataclass(frozen=True)
class HierarchicalW2Payload:
    out_features: int
    in_features: int
    r32_bits: int
    r16_bits: int
    q_packed: torch.Tensor
    parent: torch.Tensor
    r32_packed: torch.Tensor
    r16_packed: torch.Tensor
    r32_step: torch.Tensor
    r16_step: torch.Tensor
    permutation: torch.Tensor

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "q_packed": self.q_packed,
            "parent": self.parent,
            "r32_packed": self.r32_packed,
            "r16_packed": self.r16_packed,
            "r32_step": self.r32_step,
            "r16_step": self.r16_step,
            "permutation": self.permutation,
        }

    @property
    def tensor_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.tensors().values())

    @property
    def actual_bpw(self) -> float:
        return 8.0 * self.tensor_bytes / (self.out_features * self.in_features)


def make_payload(
    q: torch.Tensor,
    parent: torch.Tensor,
    r32_codes: torch.Tensor,
    r16_codes: torch.Tensor,
    *,
    r32_bits: int,
    r16_bits: int,
    r32_step: torch.Tensor,
    r16_step: torch.Tensor,
    permutation: torch.Tensor,
) -> HierarchicalW2Payload:
    rows, columns = q.shape
    if columns % GROUP_SIZE:
        raise ValueError("q width must be divisible by 128")
    groups = columns // GROUP_SIZE
    if parent.shape != (rows, groups):
        raise ValueError("parent shape mismatch")
    if r32_codes.shape != (rows, groups, 4) or r16_codes.shape != (rows, groups, 8):
        raise ValueError("relative-code shape mismatch")
    if permutation.shape != (columns,):
        raise ValueError("permutation shape mismatch")
    if set(q.unique().tolist()) - {-3, -1, 1, 3}:
        raise ValueError("q contains a value outside {-3,-1,1,3}")
    if not torch.isfinite(parent).all() or torch.any(parent <= 0):
        raise ValueError("parent scales must be finite and positive")
    for name, code, bits in (
        ("r32", r32_codes, r32_bits),
        ("r16", r16_codes, r16_bits),
    ):
        if bits not in (4, 8) or torch.any(code.to(torch.int64) >= (1 << bits)):
            raise ValueError(f"{name} codes do not fit configured bits")
    if not torch.isfinite(r32_step).all() or not torch.isfinite(r16_step).all():
        raise ValueError("relative steps must be finite")
    if torch.any(r32_step <= 0) or torch.any(r16_step <= 0):
        raise ValueError("relative steps must be positive")
    sorted_permutation = torch.sort(permutation.to(torch.int64)).values
    if not torch.equal(
        sorted_permutation,
        torch.arange(columns, dtype=torch.int64, device=permutation.device),
    ):
        raise ValueError("permutation must be a bijection")
    q_code = ((q.to(torch.int16) + 3) // 2).to(torch.uint8)
    return HierarchicalW2Payload(
        out_features=rows,
        in_features=columns,
        r32_bits=r32_bits,
        r16_bits=r16_bits,
        q_packed=pack_unsigned(q_code, 2).cpu(),
        parent=parent.to(torch.float16).cpu(),
        r32_packed=pack_unsigned(r32_codes, r32_bits).cpu(),
        r16_packed=pack_unsigned(r16_codes, r16_bits).cpu(),
        r32_step=r32_step.reshape(1).to(torch.float16).cpu(),
        r16_step=r16_step.reshape(1).to(torch.float16).cpu(),
        permutation=permutation.to(torch.int32).cpu(),
    )


def materialize_payload(
    payload: HierarchicalW2Payload,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode a quality-experiment payload into original Linear column order."""

    rows, columns = payload.out_features, payload.in_features
    groups = columns // GROUP_SIZE
    q_code = unpack_unsigned(
        payload.q_packed.to(device=device), 2, count=rows * columns, shape=(rows, columns)
    )
    q = q_code.to(torch.float32) * 2.0 - 3.0
    r32 = unpack_unsigned(
        payload.r32_packed.to(device=device),
        payload.r32_bits,
        count=rows * groups * 4,
        shape=(rows, groups, 4),
    )
    r16 = unpack_unsigned(
        payload.r16_packed.to(device=device),
        payload.r16_bits,
        count=rows * groups * 8,
        shape=(rows, groups, 8),
    )
    b = _decode_log_residual(r32, payload.r32_bits, payload.r32_step.to(device=device))
    c = _decode_log_residual(r16, payload.r16_bits, payload.r16_step.to(device=device))
    relative = torch.exp2(
        b.repeat_interleave(2, dim=-1) + c
    ).repeat_interleave(R16_WIDTH, dim=-1)
    ordered = q * payload.parent.to(device=device, dtype=torch.float32).repeat_interleave(
        GROUP_SIZE, dim=-1
    ) * relative.reshape(rows, columns)
    inverse = torch.argsort(payload.permutation.to(device=device, dtype=torch.int64))
    return ordered[:, inverse].to(dtype)


@dataclass(frozen=True)
class HierarchicalGPTQResult:
    payload: HierarchicalW2Payload
    reconstructed: torch.Tensor
    squared_error: float
    damping: float
    damp_percent: float
    dead_columns: int


@torch.no_grad()
def hierarchical_gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    r32_bits: int,
    r16_bits: int,
    group_size: int = GROUP_SIZE,
    block_size: int = GROUP_SIZE,
    desc_act: bool = True,
    damp_percent: float = 0.01,
    damp_auto_increment: float = 0.01,
    max_damp_attempts: int = 20,
    alternating_rounds: int = 3,
    step_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
) -> HierarchicalGPTQResult:
    """Run GPTQ with hierarchical W2 as the direct group projection.

    This mirrors the GPTQModel 7.4 sequential error update.  It never creates
    an ordinary W2 result and therefore cannot silently become post-hoc scale
    fitting.
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    rows, columns = weight.shape
    if group_size != GROUP_SIZE or block_size != GROUP_SIZE:
        raise ValueError("v1 requires group_size=block_size=128")
    if columns % group_size:
        raise ValueError("in_features must be divisible by group_size")
    if hessian.shape != (columns, columns):
        raise ValueError("Hessian shape mismatch")
    if damp_percent < 0 or damp_auto_increment <= 0:
        raise ValueError("invalid damping configuration")

    original = weight.to(torch.float32)
    working = original.clone()
    h = hessian.to(device=weight.device, dtype=torch.float32).clone()
    dead = h.diagonal() == 0
    h.diagonal()[dead] = 1.0
    working[:, dead] = 0.0
    permutation = (
        torch.argsort(h.diagonal(), descending=True, stable=True)
        if desc_act
        else torch.arange(columns, device=weight.device)
    )
    working = working[:, permutation]
    h = h[permutation][:, permutation]

    used_percent = damp_percent
    inverse_factor: torch.Tensor | None = None
    factor_succeeded = False
    mean_diagonal = h.diagonal().mean()
    for _ in range(max_damp_attempts):
        damped = h.clone()
        damping = mean_diagonal * used_percent
        damped.diagonal().add_(damping)
        factor, info = torch.linalg.cholesky_ex(damped)
        if int(info.max().item()) == 0:
            inverse = torch.cholesky_inverse(factor)
            inverse_factor, inverse_info = torch.linalg.cholesky_ex(inverse, upper=True)
            if int(inverse_info.max().item()) == 0:
                factor_succeeded = True
                break
        used_percent += damp_auto_increment
    if inverse_factor is None or not factor_succeeded:
        raise RuntimeError("failed to obtain a positive-definite damped Hessian")

    r32_step, r16_step = choose_linear_steps(
        working,
        r32_bits=r32_bits,
        r16_bits=r16_bits,
        multipliers=step_multipliers,
    )
    q_all = torch.empty_like(working, dtype=torch.int8)
    num_groups = columns // group_size
    parents = torch.empty((rows, num_groups), dtype=torch.float32, device=weight.device)
    r32_all = torch.empty((rows, num_groups, 4), dtype=torch.uint8, device=weight.device)
    r16_all = torch.empty((rows, num_groups, 8), dtype=torch.uint8, device=weight.device)

    for block_start in range(0, columns, block_size):
        block_stop = min(block_start + block_size, columns)
        block = working[:, block_start:block_stop].clone()
        block_q = torch.zeros_like(block)
        block_error = torch.zeros_like(block)
        for group_start in range(block_start, block_stop, group_size):
            group_stop = min(group_start + group_size, block_stop)
            if group_stop - group_start != group_size:
                raise ValueError("block boundaries must preserve complete 128-column groups")
            local_start = group_start - block_start
            local_stop = group_stop - block_start
            projection = project_hierarchical_group(
                block[:, local_start:local_stop],
                r32_bits=r32_bits,
                r16_bits=r16_bits,
                r32_step=r32_step,
                r16_step=r16_step,
                alternating_rounds=alternating_rounds,
            )
            group_index = group_start // group_size
            parents[:, group_index] = projection.parent[:, 0]
            r32_all[:, group_index] = projection.r32_codes
            r16_all[:, group_index] = projection.r16_codes
            scales = projection.effective_scale
            for column in range(group_size):
                local_column = local_start + column
                global_column = group_start + column
                current = block[:, local_column]
                q = _nearest_w2(current, scales[:, column])
                reconstructed = q * scales[:, column]
                denominator = inverse_factor[global_column, global_column]
                error = (current - reconstructed) / denominator
                block_q[:, local_column] = reconstructed
                block_error[:, local_column] = error
                q_all[:, global_column] = q.to(torch.int8)
                block[:, local_column:local_stop] -= error.unsqueeze(1) @ inverse_factor[
                    global_column, global_column:group_stop
                ].unsqueeze(0)
        working[:, block_stop:] -= block_error @ inverse_factor[
            block_start:block_stop, block_stop:
        ]

    payload = make_payload(
        q_all,
        parents,
        r32_all,
        r16_all,
        r32_bits=r32_bits,
        r16_bits=r16_bits,
        r32_step=r32_step,
        r16_step=r16_step,
        permutation=permutation,
    )
    reconstructed = materialize_payload(payload, device=weight.device, dtype=weight.dtype)
    squared_error = float((original - reconstructed.to(torch.float32)).square().sum().item())
    return HierarchicalGPTQResult(
        payload=payload,
        reconstructed=reconstructed,
        squared_error=squared_error,
        damping=float(damping.item()),
        damp_percent=used_percent,
        dead_columns=int(dead.sum().item()),
    )


def payload_from_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    out_features: int,
    in_features: int,
    r32_bits: int,
    r16_bits: int,
) -> HierarchicalW2Payload:
    required = {
        "q_packed",
        "parent",
        "r32_packed",
        "r16_packed",
        "r32_step",
        "r16_step",
        "permutation",
    }
    if set(tensors) != required:
        raise ValueError(f"payload tensor keys differ: {sorted(set(tensors) ^ required)}")
    return HierarchicalW2Payload(
        out_features=out_features,
        in_features=in_features,
        r32_bits=r32_bits,
        r16_bits=r16_bits,
        **tensors,
    )
