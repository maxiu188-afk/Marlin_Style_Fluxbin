import unittest

import torch

from fluxbin_style import (
    InputHessianAccumulator,
    TwoBaseRankOneOptimizationConfig,
    hessian_column_saliency,
    invert_hessian,
    obq_error_update,
    quantize_hybrid_two_base_obq,
    quantize_pure_two_base_obq,
    select_hessian_salient_columns,
)


class HessianOBQTests(unittest.TestCase):
    def test_accumulator_matches_normalized_two_xxt(self) -> None:
        first = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        second = torch.tensor([[[5.0, 6.0]]])
        accumulator = InputHessianAccumulator(2)
        accumulator.add(first)
        accumulator.add(second)
        flattened = torch.cat((first.reshape(-1, 2), second.reshape(-1, 2)))
        expected = 2 * flattened.T @ flattened / flattened.shape[0]
        torch.testing.assert_close(accumulator.value(), expected)
        self.assertEqual(accumulator.sample_count, 3)

    def test_inverse_hessian_uses_declared_relative_damping(self) -> None:
        hessian = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
        result = invert_hessian(hessian, damp_percent=0.1)
        expected_damping = 0.3
        expected = torch.linalg.inv(hessian + expected_damping * torch.eye(2))
        self.assertAlmostEqual(result.damping, expected_damping, places=6)
        torch.testing.assert_close(result.inverse, expected)

    def test_hessian_saliency_and_stable_group_selection(self) -> None:
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        )
        inverse = torch.diag(torch.tensor([1.0, 2.0, 1.0, 4.0, 5.0, 1.0, 7.0, 2.0]))
        observed = hessian_column_saliency(weight, inverse)
        expected = weight.to(torch.float64).square().sum(0) / inverse.diagonal().to(
            torch.float64
        ).square()
        torch.testing.assert_close(observed, expected)
        selected = select_hessian_salient_columns(
            weight,
            inverse,
            group_size=4,
            columns_per_group=2,
        )
        torch.testing.assert_close(selected, torch.tensor([[0, 2], [1, 3]]))

    def test_block_update_matches_scalar_gptq_and_reduces_quadratic_loss(self) -> None:
        inverse = torch.tensor(
            [
                [2.0, 0.4, 0.2],
                [0.4, 1.5, 0.3],
                [0.2, 0.3, 1.2],
            ]
        )
        error = torch.tensor([[0.5]])
        update = obq_error_update(error, inverse[:1, :1], inverse[:1, 1:])
        expected = error / inverse[0, 0] * inverse[0, 1:]
        torch.testing.assert_close(update, expected)

        hessian = torch.linalg.inv(inverse)
        uncompensated = torch.cat((error, torch.zeros_like(update)), dim=1)
        compensated = torch.cat((error, update), dim=1)
        loss_uncompensated = uncompensated @ hessian @ uncompensated.T
        loss_compensated = compensated @ hessian @ compensated.T
        self.assertLess(float(loss_compensated), float(loss_uncompensated))

    def test_identity_hessian_has_no_cross_group_update(self) -> None:
        error = torch.randn(3, 2)
        inverse = torch.eye(4)
        update = obq_error_update(error, inverse[:2, :2], inverse[:2, 2:])
        torch.testing.assert_close(update, torch.zeros(3, 2))

    def test_pure_and_hybrid_are_independent_and_reconstructable(self) -> None:
        torch.manual_seed(41)
        weight = torch.randn(7, 12)
        base = torch.randn(12, 12)
        hessian = base.T @ base + 0.5 * torch.eye(12)
        inverse = torch.linalg.inv(hessian)
        config = TwoBaseRankOneOptimizationConfig(
            max_iters=4,
            assignment_chunk_rows=3,
        )
        pure = quantize_pure_two_base_obq(
            weight,
            inverse,
            group_size=4,
            config=config,
        )
        hybrid = quantize_hybrid_two_base_obq(
            weight,
            inverse,
            group_size=4,
            columns_per_group=2,
            global_config=config,
            refinement_config=config,
        )
        self.assertEqual(tuple(pure.decomposition.reconstruct().shape), (7, 12))
        self.assertEqual(tuple(hybrid.decomposition.reconstruct().shape), (7, 12))
        self.assertEqual(tuple(hybrid.decomposition.selected_indices.shape), (3, 2))
        self.assertTrue(torch.isfinite(pure.decomposition.reconstruct()).all())
        self.assertTrue(torch.isfinite(hybrid.decomposition.reconstruct()).all())
        self.assertFalse(
            torch.equal(
                pure.decomposition.bases[:, :, 4:],
                hybrid.decomposition.global_decomposition.bases[:, :, 4:],
            )
        )
        global_only = hybrid.decomposition.global_decomposition.reconstruct_grouped()
        delta = hybrid.decomposition.reconstruct_grouped() - global_only
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask.scatter_(1, hybrid.decomposition.selected_indices, True)
        non_selected = ~mask.unsqueeze(0).expand(7, -1, -1)
        self.assertEqual(float(delta[non_selected].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
