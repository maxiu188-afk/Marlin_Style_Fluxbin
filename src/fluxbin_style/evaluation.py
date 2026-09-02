"""Small fail-closed helpers shared by algorithm-quality experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from fluxbin_style.packing import unpack_two_bases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sha256_token_sequences(values: Iterable[Sequence[int]]) -> str:
    """Hash signed token IDs with explicit sequence and element framing."""

    digest = hashlib.sha256()
    for sequence in values:
        digest.update(struct.pack(">Q", len(sequence)))
        for token in sequence:
            digest.update(struct.pack(">q", int(token)))
    return digest.hexdigest()


def summed_cross_entropy_fp32(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [tokens]")
    return functional.cross_entropy(logits.float(), labels, reduction="sum")


def materialize_global_two_base_weight(
    sign_codes: torch.Tensor,
    row_scales: torch.Tensor,
    column_scales: torch.Tensor,
    *,
    group_size: int,
    device: torch.device | str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize one packed global two-base rank-one weight."""

    bases, out_features, num_groups = _validate_packed_global(
        sign_codes,
        row_scales,
        column_scales,
        group_size=group_size,
        device=device,
    )
    reconstruction = torch.zeros(
        out_features,
        num_groups,
        group_size,
        device=device,
        dtype=torch.float32,
    )
    row = row_scales.to(device=device, dtype=torch.float32)
    column = column_scales.to(device=device, dtype=torch.float32)
    for base_index in range(2):
        term = bases[base_index].reshape(
            out_features,
            num_groups,
            group_size,
        ).to(dtype=torch.float32)
        term.mul_(row[base_index].unsqueeze(-1))
        term.mul_(column[base_index].unsqueeze(0))
        reconstruction.add_(term)
    return reconstruction.reshape(out_features, -1).to(dtype=output_dtype)


def materialize_hybrid_s8_weight(
    global_sign_codes: torch.Tensor,
    global_row_scales: torch.Tensor,
    global_column_scales: torch.Tensor,
    refinement_indices: torch.Tensor,
    refinement_sign_codes: torch.Tensor,
    refinement_row_scales: torch.Tensor,
    refinement_column_scales: torch.Tensor,
    *,
    group_size: int,
    columns_per_group: int,
    device: torch.device | str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize global two-base plus sparse selected-column refinement."""

    global_weight = materialize_global_two_base_weight(
        global_sign_codes,
        global_row_scales,
        global_column_scales,
        group_size=group_size,
        device=device,
        output_dtype=torch.float32,
    )
    out_features, in_features = global_weight.shape
    num_groups = in_features // group_size
    if refinement_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("refinement indices must use an integer dtype")
    if tuple(refinement_indices.shape) != (num_groups, columns_per_group):
        raise ValueError("refinement indices have the wrong shape")
    indices = refinement_indices.to(device=device, dtype=torch.int64)
    if torch.any(indices < 0) or torch.any(indices >= group_size):
        raise ValueError("refinement indices are outside the group")
    if columns_per_group > 1 and torch.any(indices[:, 1:] <= indices[:, :-1]):
        raise ValueError("refinement indices must be sorted and unique")
    expected_refinement_shape = (out_features, num_groups * columns_per_group // 4)
    if tuple(refinement_sign_codes.shape) != expected_refinement_shape:
        raise ValueError("refinement sign-code shape drifted")
    if tuple(refinement_row_scales.shape) != (2, out_features, num_groups):
        raise ValueError("refinement row-scale shape drifted")
    if tuple(refinement_column_scales.shape) != (
        2,
        num_groups,
        columns_per_group,
    ):
        raise ValueError("refinement column-scale shape drifted")
    if not refinement_row_scales.is_floating_point() or not torch.isfinite(
        refinement_row_scales
    ).all():
        raise ValueError("refinement row scales must be finite floating point")
    if not refinement_column_scales.is_floating_point() or not torch.isfinite(
        refinement_column_scales
    ).all():
        raise ValueError("refinement column scales must be finite floating point")
    bases = unpack_two_bases(refinement_sign_codes.to(device=device)).reshape(
        2,
        out_features,
        num_groups,
        columns_per_group,
    )
    row = refinement_row_scales.to(device=device, dtype=torch.float32)
    column = refinement_column_scales.to(device=device, dtype=torch.float32)
    selected = torch.zeros(
        out_features,
        num_groups,
        columns_per_group,
        device=device,
        dtype=torch.float32,
    )
    for base_index in range(2):
        term = bases[base_index].to(dtype=torch.float32)
        term.mul_(row[base_index].unsqueeze(-1))
        term.mul_(column[base_index].unsqueeze(0))
        selected.add_(term)
    grouped = global_weight.reshape(out_features, num_groups, group_size)
    scatter_indices = indices.unsqueeze(0).expand(out_features, -1, -1)
    grouped.scatter_add_(2, scatter_indices, selected)
    return grouped.reshape(out_features, in_features).to(dtype=output_dtype)


def _validate_packed_global(
    sign_codes: torch.Tensor,
    row_scales: torch.Tensor,
    column_scales: torch.Tensor,
    *,
    group_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, int, int]:
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be positive")
    if sign_codes.ndim != 2 or sign_codes.dtype != torch.uint8:
        raise ValueError("global sign codes must be a uint8 matrix")
    out_features, packed_width = sign_codes.shape
    in_features = packed_width * 4
    if in_features % group_size:
        raise ValueError("packed input width is not divisible by group_size")
    num_groups = in_features // group_size
    if tuple(row_scales.shape) != (2, out_features, num_groups):
        raise ValueError("global row-scale shape drifted")
    if tuple(column_scales.shape) != (2, num_groups, group_size):
        raise ValueError("global column-scale shape drifted")
    if not row_scales.is_floating_point() or not torch.isfinite(row_scales).all():
        raise ValueError("global row scales must be finite floating point")
    if not column_scales.is_floating_point() or not torch.isfinite(column_scales).all():
        raise ValueError("global column scales must be finite floating point")
    bases = unpack_two_bases(sign_codes.to(device=device))
    return bases, out_features, num_groups


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def error_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    if tuple(target.shape) != tuple(reconstruction.shape):
        raise ValueError("target and reconstruction shapes differ")
    target_fp32 = target.to(dtype=torch.float32)
    reconstruction_fp32 = reconstruction.to(dtype=torch.float32)
    difference = target_fp32 - reconstruction_fp32
    squared_error = difference.square().sum(dtype=torch.float64)
    target_squared_norm = target_fp32.square().sum(dtype=torch.float64)
    reconstruction_squared_norm = reconstruction_fp32.square().sum(dtype=torch.float64)
    dot = (target_fp32 * reconstruction_fp32).sum(dtype=torch.float64)
    if target_squared_norm <= 0 or reconstruction_squared_norm <= 0:
        raise ValueError("metrics require non-zero target and reconstruction norms")
    metrics = {
        "squared_error": float(squared_error.item()),
        "mse": float((squared_error / target.numel()).item()),
        "relative_frobenius_error": float(
            (squared_error.sqrt() / target_squared_norm.sqrt()).item()
        ),
        "cosine_similarity": float(
            (dot / (target_squared_norm * reconstruction_squared_norm).sqrt()).item()
        ),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "max_absolute_error": float(difference.abs().max().item()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("reconstruction metrics contain non-finite values")
    return metrics
