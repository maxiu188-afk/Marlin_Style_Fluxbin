"""Two independent rank-one-scaled binary bases for grouped Linear weights."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


NUM_BASES = 2


def _validate_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError("weight must have PyTorch Linear shape [out_features, in_features]")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains non-finite values")


def _deterministic_sign(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))


def sign_pattern_matrix(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the four two-base sign combinations in a stable bit order."""

    indices = torch.arange(1 << NUM_BASES, device=device, dtype=torch.int64)
    shifts = torch.arange(NUM_BASES, device=device, dtype=torch.int64)
    bits = indices[:, None].bitwise_right_shift(shifts[None, :]).bitwise_and(1)
    return bits.to(dtype=dtype).mul(2).sub(1)


@dataclass(frozen=True)
class TwoBaseRankOne:
    """Two binary bases with an independent rank-one magnitude per base/group.

    Shapes:

    - bases: [2, out_features, in_features]
    - row_scales: [2, out_features, num_groups]
    - column_scales: [2, num_groups, group_size]
    """

    bases: torch.Tensor
    row_scales: torch.Tensor
    column_scales: torch.Tensor
    group_size: int

    def __post_init__(self) -> None:
        if self.bases.ndim != 3 or self.bases.shape[0] != NUM_BASES:
            raise ValueError("bases must have shape [2, out_features, in_features]")
        if self.row_scales.ndim != 3 or self.row_scales.shape[0] != NUM_BASES:
            raise ValueError("row_scales must have shape [2, out_features, num_groups]")
        if self.column_scales.ndim != 3 or self.column_scales.shape[0] != NUM_BASES:
            raise ValueError("column_scales must have shape [2, num_groups, group_size]")
        if not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        _, out_features, in_features = self.bases.shape
        if in_features % self.group_size:
            raise ValueError("in_features must be divisible by group_size")
        num_groups = in_features // self.group_size
        if tuple(self.row_scales.shape) != (NUM_BASES, out_features, num_groups):
            raise ValueError("row_scales do not match bases and groups")
        if tuple(self.column_scales.shape) != (
            NUM_BASES,
            num_groups,
            self.group_size,
        ):
            raise ValueError("column_scales do not match bases and groups")
        if not torch.all((self.bases == -1) | (self.bases == 1)):
            raise ValueError("bases must contain only -1 and +1")
        if not self.row_scales.is_floating_point() or not self.column_scales.is_floating_point():
            raise TypeError("scale tensors must be floating point")
        if not torch.isfinite(self.row_scales).all() or not torch.isfinite(
            self.column_scales
        ).all():
            raise ValueError("scale tensors must be finite")
        if len({self.bases.device, self.row_scales.device, self.column_scales.device}) != 1:
            raise ValueError("bases and scale tensors must share a device")

    @property
    def out_features(self) -> int:
        return self.bases.shape[1]

    @property
    def in_features(self) -> int:
        return self.bases.shape[2]

    @property
    def num_groups(self) -> int:
        return self.in_features // self.group_size

    def grouped_bases(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        grouped = self.bases.reshape(
            NUM_BASES,
            self.out_features,
            self.num_groups,
            self.group_size,
        )
        return grouped if dtype is None else grouped.to(dtype=dtype)

    def grouped_scale_blocks(self) -> torch.Tensor:
        return self.row_scales.unsqueeze(-1) * self.column_scales.unsqueeze(1)

    def reconstruct_grouped(self) -> torch.Tensor:
        dtype = torch.promote_types(self.row_scales.dtype, self.column_scales.dtype)
        return (
            self.grouped_bases(dtype=dtype)
            * self.row_scales.to(dtype=dtype).unsqueeze(-1)
            * self.column_scales.to(dtype=dtype).unsqueeze(1)
        ).sum(dim=0)

    def reconstruct(self) -> torch.Tensor:
        return self.reconstruct_grouped().reshape(self.out_features, self.in_features)

    def squared_error(self, target: torch.Tensor) -> torch.Tensor:
        _validate_weight(target)
        if tuple(target.shape) != (self.out_features, self.in_features):
            raise ValueError("target shape does not match decomposition")
        reconstruction = self.reconstruct()
        return (target.to(dtype=reconstruction.dtype) - reconstruction).square().sum()

    def canonicalized(self) -> "TwoBaseRankOne":
        """Normalize rank-one gauges and absorb all signs into binary bases."""

        row = self.row_scales.clone()
        column = self.column_scales.clone()
        bases = self.grouped_bases().clone()
        column_rms = column.square().mean(dim=-1).sqrt()
        safe_rms = torch.where(column_rms > 0, column_rms, torch.ones_like(column_rms))
        row = row * safe_rms.unsqueeze(1)
        column = column / safe_rms.unsqueeze(-1)
        row_sign = _deterministic_sign(row)
        column_sign = _deterministic_sign(column)
        bases = bases * row_sign.unsqueeze(-1).to(dtype=bases.dtype)
        bases = bases * column_sign.unsqueeze(1).to(dtype=bases.dtype)
        return TwoBaseRankOne(
            bases=bases.reshape_as(self.bases),
            row_scales=row.abs(),
            column_scales=column.abs(),
            group_size=self.group_size,
        )


def initialize_two_base_rank_one(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
) -> TwoBaseRankOne:
    """Greedy residual initialization with row scales and unit columns."""

    _validate_weight(weight)
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("group_size must be positive and divide in_features")
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    target = weight.to(dtype=torch.float32).reshape(out_features, num_groups, group_size)
    residual = target.clone()
    bases: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    columns: list[torch.Tensor] = []
    for _ in range(NUM_BASES):
        basis = _deterministic_sign(residual)
        row = residual.abs().mean(dim=-1)
        column = torch.ones(
            (num_groups, group_size),
            dtype=torch.float32,
            device=weight.device,
        )
        residual = residual - basis * row.unsqueeze(-1)
        bases.append(basis.reshape(out_features, in_features).to(dtype=torch.int8))
        rows.append(row)
        columns.append(column)
    return TwoBaseRankOne(
        bases=torch.stack(bases),
        row_scales=torch.stack(rows),
        column_scales=torch.stack(columns),
        group_size=group_size,
    )


@dataclass(frozen=True)
class TwoBaseRankOneOptimizationConfig:
    max_iters: int = 50
    relative_tolerance: float = 1e-6
    convergence_patience: int = 3
    monotonicity_tolerance: float = 1e-6
    assignment_chunk_rows: int = 16
    denominator_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.relative_tolerance < 0 or self.monotonicity_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if self.convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive")
        if self.assignment_chunk_rows <= 0:
            raise ValueError("assignment_chunk_rows must be positive")
        if self.denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be positive")


@dataclass(frozen=True)
class TwoBaseRankOneIteration:
    iteration: int
    scale_mse: float
    mse: float
    relative_improvement: float
    assignment_changed_fraction: float
    below_tolerance_streak: int
    elapsed_seconds: float


@dataclass(frozen=True)
class TwoBaseRankOneOptimizationResult:
    decomposition: TwoBaseRankOne
    iterations: tuple[TwoBaseRankOneIteration, ...]
    converged_reason: str
    initial_mse: float
    elapsed_seconds: float


def _normalize_column_gauge(
    row: torch.Tensor,
    column: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms = column.square().mean(dim=-1).sqrt()
    safe = torch.where(rms > 0, rms, torch.ones_like(rms))
    return row * safe.unsqueeze(0), column / safe.unsqueeze(-1)


def _joint_assignment(
    target: torch.Tensor,
    row_scales: torch.Tensor,
    column_scales: torch.Tensor,
    old_bases: torch.Tensor,
    *,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    patterns = sign_pattern_matrix(device=target.device, dtype=target.dtype)
    new_bases = torch.empty_like(old_bases)
    approximation = torch.empty_like(target)
    changed = 0
    for start in range(0, target.shape[0], chunk_rows):
        stop = min(start + chunk_rows, target.shape[0])
        magnitudes = (
            row_scales[:, start:stop, :, None]
            * column_scales[:, None, :, :]
        )
        candidates = torch.einsum("pk,kogj->pogj", patterns, magnitudes)
        assignments = (
            target[start:stop].unsqueeze(0) - candidates
        ).square().argmin(dim=0)
        selected = patterns[assignments]
        chunk_bases = selected.permute(3, 0, 1, 2).to(dtype=old_bases.dtype)
        new_bases[:, start:stop] = chunk_bases
        approximation[start:stop] = (
            selected * magnitudes.permute(1, 2, 3, 0)
        ).sum(dim=-1)
        changed += int((chunk_bases != old_bases[:, start:stop]).sum().item())
    return new_bases, approximation, changed / old_bases.numel()


def optimize_two_base_rank_one(
    target: torch.Tensor,
    initial: TwoBaseRankOne,
    config: TwoBaseRankOneOptimizationConfig | None = None,
) -> TwoBaseRankOneOptimizationResult:
    """Alternate conditional rank-one LS updates and exact four-pattern assignment."""

    _validate_weight(target)
    if tuple(target.shape) != (initial.out_features, initial.in_features):
        raise ValueError("target shape does not match initial decomposition")
    if target.device != initial.bases.device:
        raise ValueError("target and initial decomposition must share a device")
    config = config or TwoBaseRankOneOptimizationConfig()
    started = time.monotonic()
    state = initial.canonicalized()
    grouped_target = target.to(dtype=torch.float32).reshape(
        initial.out_features,
        initial.num_groups,
        initial.group_size,
    )
    bases = state.grouped_bases().clone()
    row = state.row_scales.to(dtype=torch.float32).clone()
    column = state.column_scales.to(dtype=torch.float32).clone()
    approximation = (
        bases.to(dtype=torch.float32)
        * row.unsqueeze(-1)
        * column.unsqueeze(1)
    ).sum(dim=0)
    initial_mse = float((grouped_target - approximation).square().mean().item())
    previous_mse = initial_mse
    diagnostics: list[TwoBaseRankOneIteration] = []
    converged_reason = "max_iters"
    below_tolerance_streak = 0
    float_epsilon = torch.finfo(torch.float32).eps

    for iteration in range(1, config.max_iters + 1):
        iteration_started = time.monotonic()
        for basis_index in range(NUM_BASES):
            basis = bases[basis_index].to(dtype=torch.float32)
            old_contribution = (
                basis
                * row[basis_index].unsqueeze(-1)
                * column[basis_index].unsqueeze(0)
            )
            conditional_target = grouped_target - (approximation - old_contribution)
            row_design = basis * column[basis_index].unsqueeze(0)
            row_numerator = (conditional_target * row_design).sum(dim=-1)
            row_denominator = row_design.square().sum(dim=-1)
            new_row = torch.where(
                row_denominator > config.denominator_epsilon,
                row_numerator / row_denominator.clamp_min(config.denominator_epsilon),
                row[basis_index],
            )
            column_design = basis * new_row.unsqueeze(-1)
            column_numerator = (conditional_target * column_design).sum(dim=0)
            column_denominator = column_design.square().sum(dim=0)
            new_column = torch.where(
                column_denominator > config.denominator_epsilon,
                column_numerator / column_denominator.clamp_min(config.denominator_epsilon),
                column[basis_index],
            )
            new_row, new_column = _normalize_column_gauge(new_row, new_column)
            row[basis_index] = new_row
            column[basis_index] = new_column
            new_contribution = basis * new_row.unsqueeze(-1) * new_column.unsqueeze(0)
            approximation = approximation - old_contribution + new_contribution

        scale_mse = float((grouped_target - approximation).square().mean().item())
        allowed = config.monotonicity_tolerance * max(previous_mse, float_epsilon)
        if scale_mse > previous_mse + allowed:
            raise RuntimeError("rank-one scale update increased reconstruction MSE")
        bases, approximation, changed_fraction = _joint_assignment(
            grouped_target,
            row,
            column,
            bases,
            chunk_rows=config.assignment_chunk_rows,
        )
        mse = float((grouped_target - approximation).square().mean().item())
        if mse > scale_mse + allowed:
            raise RuntimeError("joint two-base assignment increased reconstruction MSE")
        relative_improvement = (previous_mse - mse) / max(previous_mse, float_epsilon)
        if relative_improvement <= config.relative_tolerance:
            below_tolerance_streak += 1
        else:
            below_tolerance_streak = 0
        diagnostics.append(
            TwoBaseRankOneIteration(
                iteration=iteration,
                scale_mse=scale_mse,
                mse=mse,
                relative_improvement=relative_improvement,
                assignment_changed_fraction=changed_fraction,
                below_tolerance_streak=below_tolerance_streak,
                elapsed_seconds=time.monotonic() - iteration_started,
            )
        )
        previous_mse = mse
        if below_tolerance_streak >= config.convergence_patience:
            converged_reason = "relative_tolerance_patience"
            break

    result = TwoBaseRankOne(
        bases=bases.reshape_as(initial.bases),
        row_scales=row,
        column_scales=column,
        group_size=initial.group_size,
    ).canonicalized()
    final_mse = float(
        (grouped_target - result.reconstruct_grouped().to(dtype=torch.float32))
        .square()
        .mean()
        .item()
    )
    allowed = config.monotonicity_tolerance * max(previous_mse, float_epsilon)
    if final_mse > previous_mse + allowed:
        raise RuntimeError("canonicalization increased reconstruction MSE")
    return TwoBaseRankOneOptimizationResult(
        decomposition=result,
        iterations=tuple(diagnostics),
        converged_reason=converged_reason,
        initial_mse=initial_mse,
        elapsed_seconds=time.monotonic() - started,
    )
