"""Versioned hybrid-only compensation repair; historical quantizers unchanged.

C = U.T @ U. After fixing a prefix, the free-subproblem inverse is
U_RR.T @ U_RR. Its block compensation coefficients reduce to
solve(U_BB, U_BR), avoiding dense Schur updates per group.
"""
from __future__ import annotations
import torch
from .hessian_obq import (
    _validate_obq_inputs, _empty_global_components, _fit_two_base_block,
    _store_group, HybridTwoBaseOBQResult, OBQGroupRecord,
)
from .two_base_rank1 import NUM_BASES, TwoBaseRankOne, TwoBaseRankOneOptimizationConfig
from .residual_refinement import SparseResidualRefinement


def conditioned_coefficients(upper: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Upper Cholesky block solve for the current conditional subproblem."""
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1]:
        raise ValueError("upper factor must be square")
    if not 0 <= start < stop <= upper.shape[0]:
        raise ValueError("invalid block bounds")
    return torch.linalg.solve_triangular(
        upper[start:stop, start:stop], upper[start:stop, stop:], upper=True,
    )


def quantize_hybrid_conditioned_v1(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    group_size: int = 128,
    columns_per_group: int = 8,
    global_config: TwoBaseRankOneOptimizationConfig | None = None,
    refinement_config: TwoBaseRankOneOptimizationConfig | None = None,
    fixed_indices: torch.Tensor,
) -> HybridTwoBaseOBQResult:
    """Conditioned compensation with externally frozen hybrid column indices."""

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
    if fixed_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("fixed indices must be integer")
    if tuple(fixed_indices.shape) != (num_groups, columns_per_group):
        raise ValueError("fixed indices shape mismatch")
    if torch.any(fixed_indices < 0) or torch.any(fixed_indices >= group_size):
        raise ValueError("fixed indices outside group")
    if torch.any(fixed_indices[:, 1:] <= fixed_indices[:, :-1]):
        raise ValueError("fixed indices must be sorted and unique")
    fixed_indices = fixed_indices.to(device=weight.device, dtype=torch.int64)
    upper = torch.linalg.cholesky(inverse_hessian.to(torch.float32), upper=True)
    records: list[OBQGroupRecord] = []
    for group_index in range(num_groups):
        start = group_index * group_size
        stop = start + group_size
        target = working[:, start:stop]
        selected_indices[group_index] = fixed_indices[group_index]
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
            coefficients = conditioned_coefficients(upper, start, stop)
            update = error @ coefficients
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
