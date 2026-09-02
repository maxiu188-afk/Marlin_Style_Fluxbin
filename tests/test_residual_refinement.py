import unittest

import torch

from fluxbin_style import (
    TwoBaseRankOneOptimizationConfig,
    initialize_two_base_rank_one,
    optimize_sparse_residual_refinement,
    optimize_two_base_rank_one,
    select_residual_columns,
)


class SparseResidualRefinementTests(unittest.TestCase):
    def test_selection_uses_top_residual_sse_per_group(self) -> None:
        residual = torch.zeros(3, 8)
        residual[:, 1] = 2
        residual[:, 3] = 4
        residual[:, 4] = 5
        residual[:, 6] = 3
        observed = select_residual_columns(
            residual,
            group_size=4,
            columns_per_group=2,
        )
        expected = torch.tensor([[1, 3], [0, 2]], dtype=torch.int64)
        torch.testing.assert_close(observed, expected)

    def test_refinement_changes_only_selected_columns(self) -> None:
        torch.manual_seed(23)
        target = torch.randn(7, 16)
        global_initial = initialize_two_base_rank_one(target, group_size=8)
        global_result = optimize_two_base_rank_one(
            target,
            global_initial,
            TwoBaseRankOneOptimizationConfig(max_iters=4, assignment_chunk_rows=3),
        )
        hybrid = optimize_sparse_residual_refinement(
            target,
            global_result.decomposition,
            columns_per_group=2,
            config=TwoBaseRankOneOptimizationConfig(
                max_iters=4,
                assignment_chunk_rows=3,
            ),
        ).decomposition
        delta = hybrid.reconstruct_grouped() - global_result.decomposition.reconstruct_grouped()
        selected_mask = torch.zeros(2, 8, dtype=torch.bool)
        selected_mask.scatter_(1, hybrid.selected_indices.cpu(), True)
        non_selected = ~selected_mask.unsqueeze(0).expand(7, -1, -1)
        self.assertEqual(float(delta.cpu()[non_selected].abs().max().item()), 0.0)

    def test_hybrid_non_regresses_global_reconstruction(self) -> None:
        torch.manual_seed(29)
        target = torch.randn(9, 24)
        global_initial = initialize_two_base_rank_one(target, group_size=6)
        global_result = optimize_two_base_rank_one(
            target,
            global_initial,
            TwoBaseRankOneOptimizationConfig(max_iters=6, assignment_chunk_rows=3),
        )
        refinement = optimize_sparse_residual_refinement(
            target,
            global_result.decomposition,
            columns_per_group=2,
            config=TwoBaseRankOneOptimizationConfig(
                max_iters=6,
                assignment_chunk_rows=3,
            ),
        )
        global_sse = (target - global_result.decomposition.reconstruct()).square().sum()
        hybrid_sse = (target - refinement.decomposition.reconstruct()).square().sum()
        self.assertLess(float(hybrid_sse), float(global_sse))
        self.assertGreater(refinement.selected_residual_energy_fraction, 0.0)
        self.assertLessEqual(refinement.selected_residual_energy_fraction, 1.0)
        self.assertEqual(tuple(refinement.decomposition.selected_indices.shape), (4, 2))
        self.assertEqual(
            tuple(refinement.decomposition.refinement_decomposition.bases.shape),
            (2, 9, 8),
        )


if __name__ == "__main__":
    unittest.main()
