"""Hessian-guided sequential two-base quantization.

This module implements the calibration-dependent parts of the FluxBin
algorithms while deliberately reusing the project's existing two-base
rank-one solver.  Pure two-base and sparse-refined quantization are separate
calls: once blockwise OBQ propagation is enabled their later working weights
are no longer shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fluxbin_style.residual_refinement import SparseResidualRefinement
from fluxbin_style.two_base_rank1 import (
    NUM_BASES,
    TwoBaseRankOne,
    TwoBaseRankOneOptimizationConfig,
    initialize_two_base_rank_one,
    optimize_two_base_rank_one,
)


def _validate_matrix(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point matrix")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


class InputHessianAccumulator:
    """Accumulate the normalized equivalent of ``H = 2 X X^T``.

    Linear activations are accepted with any leading dimensions and the last
    dimension equal to ``in_features``.  The stored matrix is divided by the
    number of activation rows.  This positive scalar normalization leaves
    Hessian saliency rankings and OBQ propagation coefficients unchanged.
    """

    def __init__(
        self,
        in_features: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        self.in_features = in_features
        self.hessian = torch.zeros(
            (in_features, in_features),
            dtype=torch.float32,
            device=device,
        )
        self.sample_count = 0

    def add(self, inputs: torch.Tensor) -> None:
        if not inputs.is_floating_point():
            raise TypeError("inputs must be floating point")
        if inputs.ndim < 2 or inputs.shape[-1] != self.in_features:
            raise ValueError("inputs must end in in_features")
        if inputs.device != self.hessian.device:
            raise ValueError("inputs and Hessian accumulator must share a device")
        flattened = inputs.detach().reshape(-1, self.in_features).to(torch.float32)
        if not torch.isfinite(flattened).all():
            raise ValueError("inputs contain non-finite values")
        added_count = flattened.shape[0]
        if added_count == 0:
            raise ValueError("inputs contain no activation rows")
        total_count = self.sample_count + added_count
        old_weight = self.sample_count / total_count
        self.hessian.mul_(old_weight)
        self.hessian.addmm_(
            flattened.T,
            flattened,
            beta=1.0,
            alpha=2.0 / total_count,
        )
        self.sample_count = total_count

    def value(self) -> torch.Tensor:
        if self.sample_count == 0:
            raise ValueError("no calibration activations were accumulated")
        return self.hessian.clone()


@dataclass(frozen=True)
class InverseHessianResult:
    inverse: torch.Tensor
    damping: float
    damp_percent: float


def invert_hessian(
    hessian: torch.Tensor,
    *,
    damp_percent: float,
) -> InverseHessianResult:
    """Damp and invert a calibration Hessian through Cholesky factorization."""

    _validate_matrix(hessian, name="hessian")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square")
    if damp_percent < 0:
        raise ValueError("damp_percent must be non-negative")
    matrix = hessian.to(torch.float32)
    if not torch.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-6):
        raise ValueError("hessian must be symmetric")
    mean_diagonal = matrix.diagonal().mean()
    if mean_diagonal <= 0:
        raise ValueError("hessian must have a positive mean diagonal")
    damping_tensor = mean_diagonal * damp_percent
    damped = matrix.clone()
    damped.diagonal().add_(damping_tensor)
    factor, info = torch.linalg.cholesky_ex(damped)
    if int(info.max().item()) != 0:
        raise ValueError("damped Hessian is not positive definite")
    inverse = torch.cholesky_inverse(factor)
    if not torch.isfinite(inverse).all():
        raise RuntimeError("inverse Hessian contains non-finite values")
    return InverseHessianResult(
        inverse=inverse,
        damping=float(damping_tensor.item()),
        damp_percent=damp_percent,
    )


def hessian_column_saliency(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
) -> torch.Tensor:
    """Return FluxBin column saliency ``sum_i W_ij^2 / Hinv_jj^2``."""

    _validate_matrix(weight, name="weight")
    _validate_matrix(inverse_hessian, name="inverse_hessian")
    if inverse_hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("inverse_hessian shape does not match weight inputs")
    diagonal = inverse_hessian.diagonal().to(torch.float64)
    if torch.any(diagonal <= 0):
        raise ValueError("inverse Hessian diagonal must be positive")
    numerator = weight.to(torch.float64).square().sum(dim=0)
    return numerator / diagonal.square()


def select_hessian_salient_columns(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    group_size: int,
    columns_per_group: int,
) -> torch.Tensor:
    """Select stable top-s Hessian-salient columns in every contiguous group."""

    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("group_size must be positive and divide in_features")
    if columns_per_group <= 0 or columns_per_group > group_size:
        raise ValueError("columns_per_group must be in [1, group_size]")
    saliency = hessian_column_saliency(weight, inverse_hessian).reshape(
        weight.shape[1] // group_size,
        group_size,
    )
    ranked = torch.argsort(saliency, dim=-1, descending=True, stable=True)
    return ranked[:, :columns_per_group].sort(dim=-1).values.to(torch.int64)


def obq_error_update(
    error: torch.Tensor,
    inverse_hessian_block: torch.Tensor,
    inverse_hessian_cross: torch.Tensor,
) -> torch.Tensor:
    """Compute the right-side weight update induced by one quantized block.

    ``error`` is ``W_block - Q_block``.  The caller subtracts the returned
    update from the unprocessed columns.
    """

    _validate_matrix(error, name="error")
    _validate_matrix(inverse_hessian_block, name="inverse_hessian_block")
    _validate_matrix(inverse_hessian_cross, name="inverse_hessian_cross")
    block_width = error.shape[1]
    if inverse_hessian_block.shape != (block_width, block_width):
        raise ValueError("inverse_hessian_block shape does not match error")
    if inverse_hessian_cross.shape[0] != block_width:
        raise ValueError("inverse_hessian_cross row count does not match block")
    coefficients = torch.linalg.solve(
        inverse_hessian_block.to(torch.float32),
        inverse_hessian_cross.to(torch.float32),
    )
    return error.to(torch.float32) @ coefficients


@dataclass(frozen=True)
class OBQGroupRecord:
    group_index: int
    start_column: int
    stop_column: int
    global_iterations: int
    global_converged_reason: str
    refinement_iterations: int | None
    refinement_converged_reason: str | None
    local_squared_error: float
    propagated_update_squared_norm: float


@dataclass(frozen=True)
class PureTwoBaseOBQResult:
    decomposition: TwoBaseRankOne
    groups: tuple[OBQGroupRecord, ...]


@dataclass(frozen=True)
class HybridTwoBaseOBQResult:
    decomposition: SparseResidualRefinement
    groups: tuple[OBQGroupRecord, ...]


def _empty_global_components(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    return (
        torch.empty(
            (NUM_BASES, out_features, in_features),
            dtype=torch.int8,
            device=weight.device,
        ),
        torch.empty(
            (NUM_BASES, out_features, num_groups),
            dtype=torch.float32,
            device=weight.device,
        ),
        torch.empty(
            (NUM_BASES, num_groups, group_size),
            dtype=torch.float32,
            device=weight.device,
        ),
    )


def _fit_two_base_block(
    target: torch.Tensor,
    config: TwoBaseRankOneOptimizationConfig,
):
    initial = initialize_two_base_rank_one(target, group_size=target.shape[1])
    return optimize_two_base_rank_one(target, initial, config)


def _store_group(
    decomposition: TwoBaseRankOne,
    *,
    group_index: int,
    start_column: int,
    stop_column: int,
    bases: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> None:
    bases[:, :, start_column:stop_column] = decomposition.bases
    rows[:, :, group_index] = decomposition.row_scales[:, :, 0]
    columns[:, group_index, :] = decomposition.column_scales[:, 0, :]


def _validate_obq_inputs(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    group_size: int,
) -> None:
    _validate_matrix(weight, name="weight")
    _validate_matrix(inverse_hessian, name="inverse_hessian")
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("group_size must be positive and divide in_features")
    if inverse_hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("inverse_hessian shape does not match weight inputs")
    if inverse_hessian.device != weight.device:
        raise ValueError("weight and inverse_hessian must share a device")


def quantize_pure_two_base_obq(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    group_size: int = 128,
    config: TwoBaseRankOneOptimizationConfig | None = None,
) -> PureTwoBaseOBQResult:
    """Quantize a weight matrix groupwise with pure two-base OBQ propagation."""

    _validate_obq_inputs(weight, inverse_hessian, group_size)
    config = config or TwoBaseRankOneOptimizationConfig()
    working = weight.to(torch.float32).clone()
    bases, rows, columns = _empty_global_components(weight, group_size=group_size)
    records: list[OBQGroupRecord] = []
    num_groups = weight.shape[1] // group_size
    for group_index in range(num_groups):
        start = group_index * group_size
        stop = start + group_size
        target = working[:, start:stop]
        fit = _fit_two_base_block(target, config)
        quantized = fit.decomposition.reconstruct().to(torch.float32)
        _store_group(
            fit.decomposition,
            group_index=group_index,
            start_column=start,
            stop_column=stop,
            bases=bases,
            rows=rows,
            columns=columns,
        )
        error = target - quantized
        update_norm = 0.0
        if stop < weight.shape[1]:
            update = obq_error_update(
                error,
                inverse_hessian[start:stop, start:stop],
                inverse_hessian[start:stop, stop:],
            )
            working[:, stop:] -= update
            update_norm = float(update.square().sum(dtype=torch.float64).item())
        records.append(
            OBQGroupRecord(
                group_index=group_index,
                start_column=start,
                stop_column=stop,
                global_iterations=len(fit.iterations),
                global_converged_reason=fit.converged_reason,
                refinement_iterations=None,
                refinement_converged_reason=None,
                local_squared_error=float(error.square().sum(dtype=torch.float64).item()),
                propagated_update_squared_norm=update_norm,
            )
        )
    return PureTwoBaseOBQResult(
        decomposition=TwoBaseRankOne(
            bases=bases,
            row_scales=rows,
            column_scales=columns,
            group_size=group_size,
        ),
        groups=tuple(records),
    )


def quantize_hybrid_two_base_obq(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    group_size: int = 128,
    columns_per_group: int = 8,
    global_config: TwoBaseRankOneOptimizationConfig | None = None,
    refinement_config: TwoBaseRankOneOptimizationConfig | None = None,
) -> HybridTwoBaseOBQResult:
    """Quantize with Hessian-selected sparse refinement and OBQ propagation."""

    _validate_obq_inputs(weight, inverse_hessian, group_size)
    if columns_per_group <= 0 or columns_per_group > group_size:
        raise ValueError("columns_per_group must be in [1, group_size]")
    global_config = global_config or TwoBaseRankOneOptimizationConfig()
    refinement_config = refinement_config or TwoBaseRankOneOptimizationConfig()
    working = weight.to(torch.float32).clone()
    global_bases, global_rows, global_columns = _empty_global_components(
        weight,
        group_size=group_size,
    )
    num_groups = weight.shape[1] // group_size
    selected_indices = torch.empty(
        (num_groups, columns_per_group),
        dtype=torch.int64,
        device=weight.device,
    )
    refinement_bases = torch.empty(
        (NUM_BASES, weight.shape[0], num_groups * columns_per_group),
        dtype=torch.int8,
        device=weight.device,
    )
    refinement_rows = torch.empty(
        (NUM_BASES, weight.shape[0], num_groups),
        dtype=torch.float32,
        device=weight.device,
    )
    refinement_columns = torch.empty(
        (NUM_BASES, num_groups, columns_per_group),
        dtype=torch.float32,
        device=weight.device,
    )
    records: list[OBQGroupRecord] = []
    for group_index in range(num_groups):
        start = group_index * group_size
        stop = start + group_size
        target = working[:, start:stop]
        # The paper computes saliency before decomposing the current group.
        # For later groups this must see the branch-specific propagated weight.
        selected_indices[group_index] = select_hessian_salient_columns(
            target,
            inverse_hessian[start:stop, start:stop],
            group_size=group_size,
            columns_per_group=columns_per_group,
        )[0]
        global_fit = _fit_two_base_block(target, global_config)
        global_quantized = global_fit.decomposition.reconstruct().to(torch.float32)
        _store_group(
            global_fit.decomposition,
            group_index=group_index,
            start_column=start,
            stop_column=stop,
            bases=global_bases,
            rows=global_rows,
            columns=global_columns,
        )
        indices = selected_indices[group_index]
        selected_residual = (target - global_quantized)[:, indices]
        refinement_fit = _fit_two_base_block(selected_residual, refinement_config)
        refinement_start = group_index * columns_per_group
        refinement_stop = refinement_start + columns_per_group
        _store_group(
            refinement_fit.decomposition,
            group_index=group_index,
            start_column=refinement_start,
            stop_column=refinement_stop,
            bases=refinement_bases,
            rows=refinement_rows,
            columns=refinement_columns,
        )
        quantized = global_quantized.clone()
        quantized[:, indices] += refinement_fit.decomposition.reconstruct().to(
            torch.float32
        )
        error = target - quantized
        update_norm = 0.0
        if stop < weight.shape[1]:
            update = obq_error_update(
                error,
                inverse_hessian[start:stop, start:stop],
                inverse_hessian[start:stop, stop:],
            )
            working[:, stop:] -= update
            update_norm = float(update.square().sum(dtype=torch.float64).item())
        records.append(
            OBQGroupRecord(
                group_index=group_index,
                start_column=start,
                stop_column=stop,
                global_iterations=len(global_fit.iterations),
                global_converged_reason=global_fit.converged_reason,
                refinement_iterations=len(refinement_fit.iterations),
                refinement_converged_reason=refinement_fit.converged_reason,
                local_squared_error=float(error.square().sum(dtype=torch.float64).item()),
                propagated_update_squared_norm=update_norm,
            )
        )
    global_decomposition = TwoBaseRankOne(
        bases=global_bases,
        row_scales=global_rows,
        column_scales=global_columns,
        group_size=group_size,
    )
    refinement_decomposition = TwoBaseRankOne(
        bases=refinement_bases,
        row_scales=refinement_rows,
        column_scales=refinement_columns,
        group_size=columns_per_group,
    )
    return HybridTwoBaseOBQResult(
        decomposition=SparseResidualRefinement(
            global_decomposition=global_decomposition,
            selected_indices=selected_indices,
            refinement_decomposition=refinement_decomposition,
        ),
        groups=tuple(records),
    )
