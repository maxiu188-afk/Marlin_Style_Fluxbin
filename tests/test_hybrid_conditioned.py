import unittest
import torch
from fluxbin_style import quantize_hybrid_two_base_obq, TwoBaseRankOneOptimizationConfig
from fluxbin_style.hybrid_conditioned import conditioned_coefficients, quantize_hybrid_conditioned_v1


class ConditionedHybridTests(unittest.TestCase):
    def test_every_group_matches_remaining_quadratic_optimum(self):
        torch.manual_seed(91)
        a = torch.randn(12, 12, dtype=torch.float64)
        h = a.T @ a + .5 * torch.eye(12, dtype=torch.float64)
        c = torch.linalg.inv(h)
        upper = torch.linalg.cholesky(c, upper=True)
        for start in (0, 3, 6):
            stop = start + 3
            current = torch.linalg.inv(h[start:, start:])
            oracle = torch.linalg.solve(current[:3, :3], current[:3, 3:])
            actual = conditioned_coefficients(upper, start, stop)
            torch.testing.assert_close(actual, oracle, rtol=1e-11, atol=1e-11)
            error = torch.randn(4, 3, dtype=torch.float64)
            delta = torch.cat([error, error @ actual], dim=1)
            # Free-coordinate gradient must vanish with the fixed group residual.
            gradient = delta @ h[start:, stop:]
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-11)
        self.assertEqual(conditioned_coefficients(upper, 9, 12).shape, (3, 0))

    def test_old_slice_fails_after_fixed_prefix(self):
        c = torch.tensor([[2., .8, .6], [.8, 1.5, .9], [.6, .9, 1.2]], dtype=torch.float64)
        h = torch.linalg.inv(c)
        old = c[1:2, 2:] / c[1, 1]
        new = conditioned_coefficients(torch.linalg.cholesky(c, upper=True), 1, 2)
        self.assertGreater(abs(float(old-new)), .04)
        self.assertLess(abs(float(torch.cat([torch.ones(1,1,dtype=torch.float64), new],1) @ h[1:,2:])), 1e-12)

    def test_identity_matches_legacy_with_identical_indices(self):
        torch.manual_seed(14)
        w = torch.randn(13,24); c = torch.eye(24)
        config = TwoBaseRankOneOptimizationConfig(max_iters=5)
        old = quantize_hybrid_two_base_obq(w,c,group_size=8,columns_per_group=1,global_config=config,refinement_config=config)
        new = quantize_hybrid_conditioned_v1(w,c,group_size=8,columns_per_group=1,global_config=config,refinement_config=config,fixed_indices=old.decomposition.selected_indices)
        torch.testing.assert_close(old.decomposition.reconstruct(),new.decomposition.reconstruct(),rtol=0,atol=0)

    def test_fixed_columns_and_inputs_preserved_for_correlated_case(self):
        torch.manual_seed(17)
        w=torch.randn(13,24); a=torch.randn(24,24); c=torch.linalg.inv(a.T@a+torch.eye(24))
        before_w=w.clone(); before_c=c.clone()
        cfg=TwoBaseRankOneOptimizationConfig(max_iters=4)
        old=quantize_hybrid_two_base_obq(w,c,group_size=8,columns_per_group=1,global_config=cfg,refinement_config=cfg)
        indices=old.decomposition.selected_indices.clone()
        new=quantize_hybrid_conditioned_v1(w,c,group_size=8,columns_per_group=1,global_config=cfg,refinement_config=cfg,fixed_indices=indices)
        torch.testing.assert_close(new.decomposition.selected_indices,indices,rtol=0,atol=0)
        torch.testing.assert_close(w,before_w,rtol=0,atol=0)
        torch.testing.assert_close(c,before_c,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(new.decomposition.reconstruct()).all())
        delta=new.decomposition.reconstruct()-new.decomposition.global_decomposition.reconstruct()
        for g in range(3):
            mask=torch.ones(8,dtype=torch.bool); mask[indices[g]]=False
            torch.testing.assert_close(delta[:,g*8:(g+1)*8][:,mask],torch.zeros(13,7),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            quantize_hybrid_conditioned_v1(w,c,group_size=8,columns_per_group=1,fixed_indices=torch.full((3,1),8))


if __name__=='__main__':unittest.main()
