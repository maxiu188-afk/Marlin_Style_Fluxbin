"""Small fail-closed helpers shared by algorithm-quality experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


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
