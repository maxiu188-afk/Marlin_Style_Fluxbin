"""Sparse residual refinement on selected columns within each weight group."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fluxbin_style.two_base_rank1 import (
    TwoBaseRankOne,
    TwoBaseRankOneOptimizationConfig,
    TwoBaseRankOneOptimizationResult,
    initialize_two_base_rank_one,
    optimize_two_base_rank_one,
)


def select_residual_columns(
    residual: torch.Tensor,
    *,
    group_size: int,
    columns_per_group: int,
) -> torch.Tensor:
    """Select stable top-S columns by residual SSE in every contiguous group."""

    if residual.ndim != 2 or not residual.is_floating_point():
        raise ValueError("residual must be a floating-point matrix")
    if not torch.isfinite(residual).all():
        raise ValueError("residual contains non-finite values")
    if group_size <= 0 or residual.shape[1] % group_size:
        raise ValueError("group_size must be positive and divide in_features")
    if columns_per_group <= 0 or columns_per_group > group_size:
        raise ValueError("columns_per_group must be in [1, group_size]")
    grouped = residual.to(dtype=torch.float32).reshape(
        residual.shape[0],
        residual.shape[1] // group_size,
        group_size,
    )
    scores = grouped.square().sum(dim=0, dtype=torch.float64)
    ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
    selected = ranked[:, :columns_per_group]
    return selected.sort(dim=-1).values.to(dtype=torch.int64)


def gather_grouped_columns(
    matrix: torch.Tensor,
    indices: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Gather [O,G,S] values from a grouped [O,K] matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [out_features, in_features]")
    if matrix.shape[1] % group_size:
        raise ValueError("group_size must divide in_features")
    num_groups = matrix.shape[1] // group_size
    if indices.ndim != 2 or indices.shape[0] != num_groups:
        raise ValueError("indices must have shape [num_groups, columns_per_group]")
    if indices.dtype != torch.int64:
        raise TypeError("indices must use torch.int64")
    if indices.device != matrix.device:
        raise ValueError("matrix and indices must share a device")
    if torch.any(indices < 0) or torch.any(indices >= group_size):
        raise ValueError("indices are outside the group")
    grouped = matrix.reshape(matrix.shape[0], num_groups, group_size)
    gather_indices = indices.unsqueeze(0).expand(matrix.shape[0], -1, -1)
    return torch.gather(grouped, dim=2, index=gather_indices)


@dataclass(frozen=True)
class SparseResidualRefinement:
    """Global two-base arm plus a second two-base arm on S columns per group."""

    global_decomposition: TwoBaseRankOne
    selected_indices: torch.Tensor
    refinement_decomposition: TwoBaseRankOne

    def __post_init__(self) -> None:
        global_value = self.global_decomposition
        refinement = self.refinement_decomposition
        if self.selected_indices.ndim != 2:
            raise ValueError("selected_indices must have shape [num_groups, S]")
        if self.selected_indices.dtype != torch.int64:
            raise TypeError("selected_indices must use torch.int64")
        if self.selected_indices.shape[0] != global_value.num_groups:
            raise ValueError("selected_indices group count does not match global arm")
        columns_per_group = self.selected_indices.shape[1]
        if columns_per_group <= 0 or columns_per_group > global_value.group_size:
            raise ValueError("invalid selected column count")
        if refinement.out_features != global_value.out_features:
            raise ValueError("global and refinement output dimensions differ")
        if refinement.num_groups != global_value.num_groups:
            raise ValueError("global and refinement group counts differ")
        if refinement.group_size != columns_per_group:
            raise ValueError("refinement group size must equal selected column count")
        devices = {
            global_value.bases.device,
            self.selected_indices.device,
            refinement.bases.device,
        }
        if len(devices) != 1:
            raise ValueError("global arm, indices, and refinement must share a device")
        sorted_indices = self.selected_indices.sort(dim=-1).values
        if not torch.equal(sorted_indices, self.selected_indices):
            raise ValueError("selected indices must be stored in ascending order")
        if torch.any(self.selected_indices < 0) or torch.any(
            self.selected_indices >= global_value.group_size
        ):
            raise ValueError("selected indices are outside the group")
        if columns_per_group > 1 and torch.any(
            self.selected_indices[:, 1:] == self.selected_indices[:, :-1]
        ):
            raise ValueError("selected indices must be unique within every group")

    @property
    def columns_per_group(self) -> int:
        return self.selected_indices.shape[1]

    def refinement_grouped(self) -> torch.Tensor:
        selected = self.refinement_decomposition.reconstruct_grouped()
        output = torch.zeros(
            (
                self.global_decomposition.out_features,
                self.global_decomposition.num_groups,
                self.global_decomposition.group_size,
            ),
            dtype=selected.dtype,
            device=selected.device,
        )
        scatter_indices = self.selected_indices.unsqueeze(0).expand(
            self.global_decomposition.out_features,
            -1,
            -1,
        )
        output.scatter_(dim=2, index=scatter_indices, src=selected)
        return output

    def reconstruct_grouped(self) -> torch.Tensor:
        return self.global_decomposition.reconstruct_grouped() + self.refinement_grouped()

    def reconstruct(self) -> torch.Tensor:
        return self.reconstruct_grouped().reshape(
            self.global_decomposition.out_features,
            self.global_decomposition.in_features,
        )


@dataclass(frozen=True)
class SparseResidualRefinementResult:
    decomposition: SparseResidualRefinement
    refinement_optimization: TwoBaseRankOneOptimizationResult
    selected_residual_energy_fraction: float


def optimize_sparse_residual_refinement(
    target: torch.Tensor,
    global_decomposition: TwoBaseRankOne,
    *,
    columns_per_group: int,
    config: TwoBaseRankOneOptimizationConfig | None = None,
) -> SparseResidualRefinementResult:
    """Fit a two-base rank-one decomposition to selected global-arm residuals."""

    if tuple(target.shape) != (
        global_decomposition.out_features,
        global_decomposition.in_features,
    ):
        raise ValueError("target shape does not match global decomposition")
    if target.device != global_decomposition.bases.device:
        raise ValueError("target and global decomposition must share a device")
    residual = target.to(dtype=torch.float32) - global_decomposition.reconstruct().to(
        dtype=torch.float32
    )
    selected_indices = select_residual_columns(
        residual,
        group_size=global_decomposition.group_size,
        columns_per_group=columns_per_group,
    )
    selected_residual = gather_grouped_columns(
        residual,
        selected_indices,
        group_size=global_decomposition.group_size,
    )
    total_energy = residual.square().sum(dtype=torch.float64)
    selected_energy = selected_residual.square().sum(dtype=torch.float64)
    if total_energy <= 0:
        raise ValueError("global decomposition has zero residual energy")
    refinement_target = selected_residual.reshape(
        global_decomposition.out_features,
        global_decomposition.num_groups * columns_per_group,
    )
    initial = initialize_two_base_rank_one(
        refinement_target,
        group_size=columns_per_group,
    )
    optimization = optimize_two_base_rank_one(
        refinement_target,
        initial,
        config,
    )
    decomposition = SparseResidualRefinement(
        global_decomposition=global_decomposition,
        selected_indices=selected_indices,
        refinement_decomposition=optimization.decomposition,
    )
    return SparseResidualRefinementResult(
        decomposition=decomposition,
        refinement_optimization=optimization,
        selected_residual_energy_fraction=float((selected_energy / total_energy).item()),
    )
