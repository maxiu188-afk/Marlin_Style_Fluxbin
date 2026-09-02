import unittest

import torch

from fluxbin_style import (
    TwoBaseRankOneOptimizationConfig,
    initialize_two_base_rank_one,
    materialize_global_two_base_weight,
    materialize_hybrid_s8_weight,
    optimize_sparse_residual_refinement,
    optimize_two_base_rank_one,
    pack_two_bases,
)


class PackedMaterializationTests(unittest.TestCase):
    def test_global_and_hybrid_match_reference_reconstruction(self) -> None:
        torch.manual_seed(37)
        target = torch.randn(9, 32)
        global_result = optimize_two_base_rank_one(
            target,
            initialize_two_base_rank_one(target, group_size=8),
            TwoBaseRankOneOptimizationConfig(max_iters=4, assignment_chunk_rows=3),
        )
        refinement = optimize_sparse_residual_refinement(
            target,
            global_result.decomposition,
            columns_per_group=2,
            config=TwoBaseRankOneOptimizationConfig(
                max_iters=4,
                assignment_chunk_rows=3,
            ),
        ).decomposition
        global_value = refinement.global_decomposition
        sparse = refinement.refinement_decomposition
        materialized_global = materialize_global_two_base_weight(
            pack_two_bases(global_value.bases),
            global_value.row_scales,
            global_value.column_scales,
            group_size=8,
            device="cpu",
            output_dtype=torch.float32,
        )
        materialized_hybrid = materialize_hybrid_s8_weight(
            pack_two_bases(global_value.bases),
            global_value.row_scales,
            global_value.column_scales,
            refinement.selected_indices.to(torch.int16),
            pack_two_bases(sparse.bases),
            sparse.row_scales,
            sparse.column_scales,
            group_size=8,
            columns_per_group=2,
            device="cpu",
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(materialized_global, global_value.reconstruct())
        torch.testing.assert_close(materialized_hybrid, refinement.reconstruct())


if __name__ == "__main__":
    unittest.main()
