"""Shared accounting and reporting for the Qwen3-8B W3/QBB quality study.

This module is deliberately backend-free.  It counts the persistent tensors
needed to reconstruct the 252 target Linear weights and renders the final
rate--distortion table; it does not compile or benchmark a CUDA kernel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .qwen3 import QWEN3_LINEAR_MODULES, expected_qwen3_linear_shape


EXPECTED_LAYERS = 36
EXPECTED_LINEARS = 252
EXPECTED_WEIGHTS = 6_945_767_424
SCALE_SUFFIXES = (
    "global_row_scales",
    "global_column_scales",
    "refinement_row_scales",
    "refinement_column_scales",
)
FIXED_QBB_SUFFIXES = (
    "global_sign_codes",
    "refinement_indices",
    "refinement_sign_codes",
)


def tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def target_shapes(model_config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(expected_qwen3_linear_shape(dict(model_config), module))
        for module in QWEN3_LINEAR_MODULES
    )


def target_weight_count(model_config: Mapping[str, Any], *, layers: int = EXPECTED_LAYERS) -> int:
    if layers <= 0:
        raise ValueError("layers must be positive")
    return layers * sum(out_features * in_features for out_features, in_features in target_shapes(model_config))


def analytical_qbb_storage(
    model_config: Mapping[str, Any],
    *,
    scale_bytes: int,
    index_bytes: int = 2,
    include_lookup: bool = False,
    layers: int = EXPECTED_LAYERS,
) -> dict[str, Any]:
    """Count the seven persistent QBB payload tensors, excluding file headers."""

    if scale_bytes not in (2, 4):
        raise ValueError("QBB scales must use FP16 or FP32 storage")
    if index_bytes not in (2, 4, 8):
        raise ValueError("indices must use 16, 32 or 64 bits")
    categories = {
        "global_sign_codes": 0,
        "global_row_scales": 0,
        "global_column_scales": 0,
        "refinement_sign_codes": 0,
        "refinement_row_scales": 0,
        "refinement_column_scales": 0,
        "refinement_indices": 0,
        "lookup": 0,
    }
    for out_features, in_features in target_shapes(model_config):
        if in_features % 128:
            raise ValueError("QBB target width is not divisible by group size")
        groups = in_features // 128
        categories["global_sign_codes"] += out_features * in_features // 4
        categories["global_row_scales"] += 2 * out_features * groups * scale_bytes
        categories["global_column_scales"] += 2 * groups * 128 * scale_bytes
        categories["refinement_sign_codes"] += out_features * groups * 2
        categories["refinement_row_scales"] += 2 * out_features * groups * scale_bytes
        categories["refinement_column_scales"] += 2 * groups * 8 * scale_bytes
        categories["refinement_indices"] += groups * 8 * index_bytes
        if include_lookup:
            categories["lookup"] += groups * 128 * index_bytes
    categories = {name: value * layers for name, value in categories.items()}
    total = sum(categories.values())
    weights = target_weight_count(model_config, layers=layers)
    return {
        "categories": categories,
        "persistent_tensor_bytes": total,
        "quantized_weight_count": weights,
        "effective_bits_per_weight": total * 8 / weights,
        "includes_lookup": include_lookup,
        "scale_bytes": scale_bytes,
        "index_bytes": index_bytes,
    }


def analytical_gptq_storage(
    model_config: Mapping[str, Any],
    *,
    bits: int = 3,
    group_size: int = 128,
    scale_bytes: int = 2,
    layers: int = EXPECTED_LAYERS,
) -> dict[str, Any]:
    """Count the semantic symmetric GPTQ representation before library padding.

    Constant symmetric zero points, packed-container padding and desc_act
    permutations are reported from the serialized GPTQModel artifact instead of
    being guessed here.
    """

    if bits != 3 or group_size != 128 or scale_bytes != 2:
        raise ValueError("the primary baseline is fixed to symmetric W3 g128 with FP16 scales")
    weights = target_weight_count(model_config, layers=layers)
    code_bits = weights * bits
    if code_bits % 8:
        raise ValueError("target inventory does not pack into whole bytes")
    scale_count = 0
    for out_features, in_features in target_shapes(model_config):
        if in_features % group_size:
            raise ValueError("GPTQ target width is not divisible by group size")
        scale_count += out_features * (in_features // group_size)
    scale_count *= layers
    categories = {
        "packed_weight_codes": code_bits // 8,
        "fp16_scales": scale_count * scale_bytes,
    }
    total = sum(categories.values())
    return {
        "categories": categories,
        "persistent_tensor_bytes": total,
        "quantized_weight_count": weights,
        "effective_bits_per_weight": total * 8 / weights,
        "excluded_until_serialized": [
            "constant_or_stored_zero_points",
            "desc_act_g_idx_or_permutation",
            "packing_padding",
            "safetensors_headers",
        ],
    }


def classify_gptq_tensor(name: str) -> str | None:
    for suffix, category in (
        (".qweight", "packed_weight_codes"),
        (".scales", "scales"),
        (".qzeros", "zero_points"),
        (".g_idx", "group_indices_or_permutation"),
    ):
        if name.endswith(suffix):
            return category
    return None


def summarize_tensor_storage(tensors: Mapping[str, torch.Tensor], *, gptq: bool = False) -> dict[str, Any]:
    categories: dict[str, int] = {}
    uncategorized: list[str] = []
    for name, value in tensors.items():
        if gptq:
            category = classify_gptq_tensor(name)
        elif name.endswith(SCALE_SUFFIXES):
            category = next(suffix for suffix in SCALE_SUFFIXES if name.endswith(suffix))
        elif name.endswith(FIXED_QBB_SUFFIXES):
            category = next(suffix for suffix in FIXED_QBB_SUFFIXES if name.endswith(suffix))
        else:
            category = None
        if category is None:
            uncategorized.append(name)
            continue
        categories[category] = categories.get(category, 0) + tensor_nbytes(value)
    return {
        "categories": dict(sorted(categories.items())),
        "tensor_bytes": sum(categories.values()),
        "uncategorized": sorted(uncategorized),
    }


def validate_arm_metrics(arms: Mapping[str, Mapping[str, Any]], expected_transitions: int) -> None:
    expected = {"bf16", "qbb_current", "gptq_w3_g128_sym", "qbb_fp16_scales"}
    if set(arms) != expected:
        raise ValueError(f"rate-distortion arm set drifted: {sorted(arms)}")
    for name, record in arms.items():
        ppl = record.get("perplexity")
        if (
            record.get("scored_transition_count") != expected_transitions
            or record.get("metrics_valid") is not True
            or not isinstance(ppl, (int, float))
            or not math.isfinite(ppl)
            or ppl <= 0
        ):
            raise ValueError(f"invalid PPL metrics for {name}")


def build_summary_rows(
    arms: Mapping[str, Mapping[str, Any]],
    storage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_arm_metrics(arms, 298_862)
    bf16 = float(arms["bf16"]["perplexity"])
    qbb = float(arms["qbb_current"]["perplexity"])
    labels = {
        "bf16": "Arm 0 — BF16",
        "qbb_current": "Arm 1 — current QBB",
        "gptq_w3_g128_sym": "Arm 2 — GPTQ W3A16 g128 sym",
        "qbb_fp16_scales": "Arm 3 — QBB FP16 scales",
    }
    rows = []
    for name in ("bf16", "qbb_current", "gptq_w3_g128_sym", "qbb_fp16_scales"):
        ppl = float(arms[name]["perplexity"])
        item = storage[name]
        rows.append(
            {
                "arm": name,
                "label": labels[name],
                "nominal_bits": item["nominal_bits"],
                "effective_bits_per_weight": item["effective_bits_per_weight"],
                "serialized_weight_storage_bytes": item["serialized_weight_storage_bytes"],
                "linear_coverage": item["linear_coverage"],
                "perplexity": ppl,
                "delta_ppl_vs_bf16": ppl - bf16,
                "delta_ppl_vs_current_qbb": ppl - qbb,
            }
        )
    return rows


def render_summary_markdown(rows: list[Mapping[str, Any]], *, status: str) -> str:
    lines = [
        "# Qwen3-8B uniform W3 vs QBB rate--distortion",
        "",
        f"Status: `{status}`.",
        "",
        "All PPL values use the same frozen 146 x 2048 WikiText-2 test blocks,",
        "298,862 prediction positions, BF16 activation/evaluation path, and exact",
        "36-layer / 252-Linear coverage.",
        "",
        "| Arm | Nominal bits | Effective bits/weight | Serialized weight storage | Linear coverage | PPL | Delta PPL vs BF16 | Delta PPL vs current QBB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {nominal_bits} | {effective_bits_per_weight:.6f} | "
            "{serialized_weight_storage_bytes} B | {linear_coverage} | {perplexity:.9f} | "
            "{delta_ppl_vs_bf16:+.9f} | {delta_ppl_vs_current_qbb:+.9f} |".format(**row)
        )
    lines += [
        "",
        "The Case A/B/C decision is intentionally left for effect-size review after",
        "all four arms pass provenance, coverage, decode and evaluator gates; no small",
        "automatic threshold is encoded in the runner.",
        "",
    ]
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
