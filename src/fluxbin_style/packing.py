"""Compact, lossless storage for two binary bases.

This is an algorithm-artifact format, not a deployment-kernel layout.
"""

from __future__ import annotations

import torch


TWO_BASE_PACKED_FORMAT = "fluxbin-two-base-interleaved-2bit-v1"


def pack_two_bases(bases: torch.Tensor) -> torch.Tensor:
    """Pack four K positions from two {-1,+1} bases into each uint8 code."""

    if bases.ndim != 3 or bases.shape[0] != 2:
        raise ValueError("bases must have shape [2, out_features, in_features]")
    if bases.shape[-1] % 4:
        raise ValueError("in_features must be divisible by four")
    if not torch.all((bases == -1) | (bases == 1)):
        raise ValueError("bases must contain only -1 and +1")
    bits = (bases > 0).to(dtype=torch.uint8).reshape(
        2,
        bases.shape[1],
        bases.shape[2] // 4,
        4,
    )
    shifts = torch.arange(4, device=bases.device, dtype=torch.uint8).mul(2)
    base0 = torch.bitwise_left_shift(bits[0], shifts)
    base1 = torch.bitwise_left_shift(bits[1], shifts + 1)
    return torch.bitwise_or(base0, base1).sum(dim=-1).to(dtype=torch.uint8)


def unpack_two_bases(codes: torch.Tensor) -> torch.Tensor:
    """Decode the artifact format into int8 bases shaped [2,O,K]."""

    if codes.ndim != 2 or codes.dtype != torch.uint8:
        raise ValueError("codes must be a uint8 tensor shaped [out_features, K/4]")
    shifts = torch.arange(4, device=codes.device, dtype=torch.uint8).mul(2)
    expanded = codes.unsqueeze(-1)
    base0 = torch.bitwise_and(torch.bitwise_right_shift(expanded, shifts), 1)
    base1 = torch.bitwise_and(torch.bitwise_right_shift(expanded, shifts + 1), 1)
    bits = torch.stack((base0, base1), dim=0)
    return bits.mul(2).sub(1).to(dtype=torch.int8).reshape(
        2,
        codes.shape[0],
        codes.shape[1] * 4,
    )
