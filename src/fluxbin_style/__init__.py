"""Two-base rank-one binary quantization reference package."""

from fluxbin_style.evaluation import (
    atomic_json,
    error_metrics,
    materialize_global_two_base_weight,
    materialize_hybrid_s8_weight,
    sha256_file,
    sha256_token_sequences,
    summed_cross_entropy_fp32,
    tensor_sha256,
)
from fluxbin_style.packing import (
    TWO_BASE_PACKED_FORMAT,
    pack_two_bases,
    unpack_two_bases,
)
from fluxbin_style.qwen3 import (
    QWEN3_LINEAR_MODULES,
    build_qwen3_linear_inventory,
    expected_qwen3_linear_shape,
    parse_qwen3_linear_name,
    qwen3_linear_weight_names,
)
from fluxbin_style.residual_refinement import (
    SparseResidualRefinement,
    SparseResidualRefinementResult,
    gather_grouped_columns,
    optimize_sparse_residual_refinement,
    select_residual_columns,
)
from fluxbin_style.two_base_rank1 import (
    NUM_BASES,
    TwoBaseRankOne,
    TwoBaseRankOneIteration,
    TwoBaseRankOneOptimizationConfig,
    TwoBaseRankOneOptimizationResult,
    initialize_two_base_rank_one,
    optimize_two_base_rank_one,
    sign_pattern_matrix,
)

__all__ = [
    "NUM_BASES",
    "QWEN3_LINEAR_MODULES",
    "SparseResidualRefinement",
    "SparseResidualRefinementResult",
    "TwoBaseRankOne",
    "TwoBaseRankOneIteration",
    "TwoBaseRankOneOptimizationConfig",
    "TwoBaseRankOneOptimizationResult",
    "TWO_BASE_PACKED_FORMAT",
    "atomic_json",
    "build_qwen3_linear_inventory",
    "error_metrics",
    "expected_qwen3_linear_shape",
    "gather_grouped_columns",
    "initialize_two_base_rank_one",
    "materialize_global_two_base_weight",
    "materialize_hybrid_s8_weight",
    "optimize_two_base_rank_one",
    "optimize_sparse_residual_refinement",
    "pack_two_bases",
    "parse_qwen3_linear_name",
    "qwen3_linear_weight_names",
    "sha256_file",
    "sha256_token_sequences",
    "sign_pattern_matrix",
    "select_residual_columns",
    "tensor_sha256",
    "summed_cross_entropy_fp32",
    "unpack_two_bases",
]
