import unittest

import torch

from fluxbin_style import (
    NUM_BASES,
    TwoBaseRankOne,
    TwoBaseRankOneOptimizationConfig,
    initialize_two_base_rank_one,
    optimize_two_base_rank_one,
    sign_pattern_matrix,
)


class TwoBaseRankOneTests(unittest.TestCase):
    def test_pattern_table_has_exactly_four_unique_sign_pairs(self) -> None:
        patterns = sign_pattern_matrix()
        self.assertEqual(tuple(patterns.shape), (4, 2))
        self.assertEqual(torch.unique(patterns, dim=0).shape[0], 4)
        self.assertTrue(torch.all((patterns == -1) | (patterns == 1)))

    def test_initializer_is_two_step_greedy_residual(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(5, 16)
        observed = initialize_two_base_rank_one(weight, group_size=4)
        grouped = weight.reshape(5, 4, 4)
        residual = grouped.clone()
        expected = torch.zeros_like(grouped)
        for _ in range(NUM_BASES):
            basis = torch.where(residual < 0, -1.0, 1.0)
            row = residual.abs().mean(dim=-1)
            expected += basis * row.unsqueeze(-1)
            residual -= basis * row.unsqueeze(-1)
        torch.testing.assert_close(observed.reconstruct_grouped(), expected)
        self.assertEqual(tuple(observed.bases.shape), (2, 5, 16))
        self.assertEqual(tuple(observed.row_scales.shape), (2, 5, 4))
        self.assertEqual(tuple(observed.column_scales.shape), (2, 4, 4))

    def test_each_scale_block_has_rank_at_most_one(self) -> None:
        torch.manual_seed(11)
        value = initialize_two_base_rank_one(torch.randn(6, 12), group_size=4)
        blocks = value.grouped_scale_blocks()
        for basis in range(NUM_BASES):
            for group in range(3):
                self.assertLessEqual(
                    int(torch.linalg.matrix_rank(blocks[basis, :, group]).item()),
                    1,
                )

    def test_canonicalization_preserves_reconstruction(self) -> None:
        torch.manual_seed(17)
        bases = torch.where(torch.randn(2, 3, 8) < 0, -1, 1).to(torch.int8)
        value = TwoBaseRankOne(
            bases=bases,
            row_scales=torch.randn(2, 3, 2),
            column_scales=torch.randn(2, 2, 4),
            group_size=4,
        )
        canonical = value.canonicalized()
        torch.testing.assert_close(canonical.reconstruct(), value.reconstruct())
        self.assertTrue(torch.all(canonical.row_scales >= 0))
        self.assertTrue(torch.all(canonical.column_scales >= 0))

    def test_solver_is_monotonic_and_improves_greedy_parent(self) -> None:
        torch.manual_seed(13)
        target = torch.randn(9, 24)
        initial = initialize_two_base_rank_one(target, group_size=6)
        result = optimize_two_base_rank_one(
            target,
            initial,
            TwoBaseRankOneOptimizationConfig(
                max_iters=12,
                convergence_patience=3,
                assignment_chunk_rows=3,
            ),
        )
        previous = result.initial_mse
        for item in result.iterations:
            self.assertLessEqual(item.scale_mse, previous + 1e-6)
            self.assertLessEqual(item.mse, item.scale_mse + 1e-6)
            previous = item.mse
        self.assertLessEqual(
            float(result.decomposition.squared_error(target)),
            float(initial.squared_error(target)) + 1e-5,
        )

    def test_exact_two_base_matrix_remains_exact(self) -> None:
        patterns = sign_pattern_matrix(dtype=torch.int8)
        bases = patterns.T.reshape(2, 1, 4)
        initial = TwoBaseRankOne(
            bases=bases,
            row_scales=torch.tensor([0.75, 0.25]).reshape(2, 1, 1),
            column_scales=torch.ones(2, 1, 4),
            group_size=4,
        )
        target = initial.reconstruct()
        result = optimize_two_base_rank_one(
            target,
            initial,
            TwoBaseRankOneOptimizationConfig(max_iters=5, assignment_chunk_rows=1),
        )
        torch.testing.assert_close(
            result.decomposition.reconstruct(),
            target,
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
