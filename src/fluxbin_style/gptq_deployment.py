"""Exact GPTQModel-7.4 W3 unpacking and the M=1 planar LUT layout.

The source checkpoint uses GPTQ's historical 10-1-10-1-10 packing across
three int32 words for every 32 logical 3-bit values.  The deployment layout
stores three byte-addressed bit planes after sorting input columns by ``g_idx``.
The high plane is inverted so a symmetric zero point of four becomes the
signed expression ``b0 + 2*b1 - 4*not_b2`` in the CUDA kernel.

All conversion is offline and lossless.  This module has no GPTQModel import so
the accepted raw safetensors can be inspected in the ordinary project runtime.
"""
from __future__ import annotations

from typing import Mapping

import torch


FORMAT = "gptq-w3-g128-sym-planar-m1-v1"
GROUP_SIZE = 128
BITS = 3
ZERO = 4
FIELDS = ("planes", "scales", "perm")

_WF = torch.tensor(
    (
        (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0),
        (0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31),
        (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0),
    ),
    dtype=torch.int64,
)


def _require_cpu_int32(name: str, value: torch.Tensor, ndim: int) -> None:
    if value.device.type != "cpu" or value.dtype != torch.int32 or value.ndim != ndim:
        raise ValueError(f"{name} must be CPU int32 rank {ndim}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _unpack_10_1_10_1_10(words: torch.Tensor, *, packed_axis: int) -> torch.Tensor:
    """Unpack GPTQ's three-int32 representation along rows or columns."""
    if packed_axis == 0:
        if words.shape[0] % 3:
            raise ValueError("packed row count must be divisible by three")
        blocks = words.reshape(words.shape[0] // 3, 3, 1, words.shape[1])
        shifts = _WF.reshape(1, 3, 12, 1)
    elif packed_axis == 1:
        if words.shape[1] % 3:
            raise ValueError("packed column count must be divisible by three")
        blocks = words.reshape(words.shape[0], words.shape[1] // 3, 3, 1)
        shifts = _WF.reshape(1, 1, 3, 12)
    else:
        raise ValueError("packed_axis must be zero or one")

    # int64 avoids backend-specific signed-int32 shift behavior.  Masking after
    # the arithmetic shift recovers the original unsigned bit fields.
    unpacked = torch.bitwise_right_shift(blocks.to(torch.int64), shifts).bitwise_and_(7).clone()
    if packed_axis == 0:
        unpacked[:, 0, 10] = (unpacked[:, 0, 10] & 3) | ((unpacked[:, 1, 0] << 2) & 4)
        unpacked[:, 1, 11] = (unpacked[:, 1, 11] & 1) | ((unpacked[:, 2, 0] << 1) & 6)
        return torch.cat(
            (unpacked[:, 0, :11], unpacked[:, 1, 1:12], unpacked[:, 2, 1:11]), dim=1
        ).reshape(-1, words.shape[1]).to(torch.uint8)

    unpacked[:, :, 0, 10] = (unpacked[:, :, 0, 10] & 3) | ((unpacked[:, :, 1, 0] << 2) & 4)
    unpacked[:, :, 1, 11] = (unpacked[:, :, 1, 11] & 1) | ((unpacked[:, :, 2, 0] << 1) & 6)
    return torch.cat(
        (unpacked[:, :, 0, :11], unpacked[:, :, 1, 1:12], unpacked[:, :, 2, 1:11]), dim=2
    ).reshape(words.shape[0], -1).to(torch.uint8)


def unpack_gptq_w3_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Return logical unsigned W3 codes with shape ``[K, O]``."""
    _require_cpu_int32("qweight", qweight, 2)
    return _unpack_10_1_10_1_10(qweight, packed_axis=0)


def unpack_gptq_w3_qzeros(qzeros: torch.Tensor, *, qzero_format: int) -> torch.Tensor:
    """Return logical zero points ``[G, O]``.

    Raw ``FORMAT.GPTQ`` safetensors use qzero format 1, which stores logical
    zero points minus one.  GPTQModel's Torch backend converts these to format 2
    during load.  The format is therefore explicit and never guessed.
    """
    _require_cpu_int32("qzeros", qzeros, 2)
    if qzero_format not in (1, 2):
        raise ValueError("qzero_format must be 1 (raw GPTQ) or 2 (loaded GPTQ_V2)")
    zeros = _unpack_10_1_10_1_10(qzeros, packed_axis=1)
    if qzero_format == 1:
        zeros = ((zeros.to(torch.int16) + 1) & 7).to(torch.uint8)
    return zeros


def _pack_bytes(bits: torch.Tensor) -> torch.Tensor:
    if bits.dtype != torch.uint8 or bits.shape[-1] != 128:
        raise ValueError("bit tensor must be uint8 with a 128-element final dimension")
    chunks = bits.reshape(*bits.shape[:-1], 16, 8).to(torch.int16)
    shifts = torch.arange(8, dtype=torch.int16, device=bits.device)
    return torch.sum(chunks << shifts, dim=-1).to(torch.uint8)


def _unpack_bytes(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.uint8 or values.shape[-1] != 16:
        raise ValueError("packed planes must be uint8 with a 16-byte final dimension")
    shifts = torch.arange(8, dtype=torch.int16, device=values.device)
    return ((values.to(torch.int16).unsqueeze(-1) >> shifts) & 1).to(torch.uint8).flatten(-2)


def inspect_gptq_checkpoint(
    tensors: Mapping[str, torch.Tensor], *, qzero_format: int = 1
) -> dict[str, object]:
    required = {"qweight", "qzeros", "scales", "g_idx"}
    if set(tensors) != required:
        raise ValueError(f"expected exactly {sorted(required)}")
    qweight, qzeros = tensors["qweight"], tensors["qzeros"]
    scales, g_idx = tensors["scales"], tensors["g_idx"]
    _require_cpu_int32("qweight", qweight, 2)
    _require_cpu_int32("qzeros", qzeros, 2)
    _require_cpu_int32("g_idx", g_idx, 1)
    if scales.device.type != "cpu" or scales.dtype != torch.float16 or scales.ndim != 2:
        raise ValueError("scales must be contiguous CPU float16 [G,O]")
    if not scales.is_contiguous() or not torch.isfinite(scales).all():
        raise ValueError("scales must be contiguous and finite")

    codes = unpack_gptq_w3_qweight(qweight)
    zeros = unpack_gptq_w3_qzeros(qzeros, qzero_format=qzero_format)
    k, o = codes.shape
    if k % GROUP_SIZE or k > 32767:
        raise ValueError("v1 requires K divisible by 128 and representable by int16")
    groups = k // GROUP_SIZE
    if tuple(scales.shape) != (groups, o) or tuple(zeros.shape) != (groups, o):
        raise ValueError("qzero/scale shape does not match qweight")
    if tuple(g_idx.shape) != (k,) or int(g_idx.min()) != 0 or int(g_idx.max()) != groups - 1:
        raise ValueError("g_idx shape or range drifted")
    counts = torch.bincount(g_idx.to(torch.int64), minlength=groups)
    if not torch.equal(counts, torch.full((groups,), GROUP_SIZE, dtype=torch.int64)):
        raise ValueError("each g_idx group must contain exactly 128 input columns")
    perm = torch.argsort(g_idx, stable=True)
    expected_groups = torch.arange(groups, dtype=torch.int32).repeat_interleave(GROUP_SIZE)
    if not torch.equal(g_idx[perm], expected_groups):
        raise ValueError("sorted g_idx is not canonical group128")
    unique_zeros = [int(value) for value in torch.unique(zeros).tolist()]
    return {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "qweight_shape": list(qweight.shape),
        "qzeros_shape": list(qzeros.shape),
        "scales_shape": list(scales.shape),
        "g_idx_shape": list(g_idx.shape),
        "in_features": k,
        "out_features": o,
        "groups": groups,
        "qzero_format": qzero_format,
        "decoded_zero_unique": unique_zeros,
        "permutation_identity": bool(torch.equal(perm, torch.arange(k))),
    }


def validate_planar_w3(
    layout: Mapping[str, torch.Tensor], *, check_values: bool = True
) -> tuple[int, int]:
    """Validate the deployment ABI.

    ``check_values=False`` is the allocation-free/runtime form.  The full
    finite-scale and permutation-bijection checks belong at artifact load time;
    running them in the hot wrapper would launch extra CUDA kernels before a
    graph capture and would contaminate the formal total-latency contract.
    """
    if set(layout) != set(FIELDS):
        raise ValueError(f"expected exactly {FIELDS}")
    planes, scales, perm = (layout[name] for name in FIELDS)
    if planes.dtype != torch.uint8 or planes.ndim != 4 or planes.shape[1] != 3 or planes.shape[3] != 16:
        raise ValueError("planes must be uint8 [G,3,O,16]")
    groups, _, out_features, _ = planes.shape
    if scales.dtype != torch.bfloat16 or tuple(scales.shape) != (groups, out_features):
        raise ValueError("deployment scales must be bfloat16 [G,O]")
    in_features = groups * GROUP_SIZE
    if in_features > 32767:
        raise ValueError("v1 perm indices must be representable by int16")
    if perm.dtype != torch.int16 or tuple(perm.shape) != (in_features,):
        raise ValueError("perm must be int16 [K]")
    if any(value.device != planes.device or not value.is_contiguous() for value in layout.values()):
        raise ValueError("layout tensors must be contiguous on one device")
    if check_values:
        if not torch.isfinite(scales).all():
            raise ValueError("scales must be finite")
        p64 = perm.to(torch.int64)
        if (
            int(p64.min()) != 0
            or int(p64.max()) != in_features - 1
            or torch.unique(p64).numel() != in_features
        ):
            raise ValueError("perm must be a bijection over K")
    return out_features, in_features


def convert_gptq_w3_to_planar(
    tensors: Mapping[str, torch.Tensor], *, qzero_format: int = 1
) -> dict[str, torch.Tensor]:
    facts = inspect_gptq_checkpoint(tensors, qzero_format=qzero_format)
    if facts["decoded_zero_unique"] != [ZERO]:
        raise ValueError(
            f"v1 constant-zero fast path requires decoded zero=4, got {facts['decoded_zero_unique']}"
        )
    codes = unpack_gptq_w3_qweight(tensors["qweight"])
    g_idx = tensors["g_idx"]
    perm64 = torch.argsort(g_idx, stable=True)
    groups, out_features = tensors["scales"].shape
    sorted_codes = codes[perm64].reshape(groups, GROUP_SIZE, out_features).permute(0, 2, 1)
    b0 = sorted_codes & 1
    b1 = (sorted_codes >> 1) & 1
    # The stored high plane is inverted so q-4 = b0 + 2*b1 - 4*stored_b2.
    b2_inverted = 1 - ((sorted_codes >> 2) & 1)
    layout = {
        "planes": torch.stack((_pack_bytes(b0), _pack_bytes(b1), _pack_bytes(b2_inverted)), dim=1)
        .contiguous(),
        # GPTQModel 7.4 reloads this checkpoint with dtype=bfloat16, so its
        # canonical retained decoder casts raw FP16 scales before dequantizing.
        # Store that exact two-byte deployment value; keeping raw FP16 here
        # would differ from the accepted decoded-BF16 oracle.
        "scales": tensors["scales"].to(torch.bfloat16).contiguous(),
        "perm": perm64.to(torch.int16).contiguous(),
    }
    validate_planar_w3(layout)
    return layout


def decode_planar_w3_codes(
    layout: Mapping[str, torch.Tensor], *, original_order: bool = True
) -> torch.Tensor:
    """Return logical unsigned codes as ``[K,O]``."""
    out_features, in_features = validate_planar_w3(layout)
    planes = layout["planes"]
    b0 = _unpack_bytes(planes[:, 0])
    b1 = _unpack_bytes(planes[:, 1])
    b2 = 1 - _unpack_bytes(planes[:, 2])
    sorted_codes = (b0 + (b1 << 1) + (b2 << 2)).permute(0, 2, 1).reshape(in_features, out_features)
    if not original_order:
        return sorted_codes.contiguous()
    restored = torch.empty_like(sorted_codes)
    restored[layout["perm"].to(torch.int64)] = sorted_codes
    return restored


def restore_planar_w3(
    layout: Mapping[str, torch.Tensor], *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Restore ``[O,K]`` weights with GPTQModel's loaded-BF16 semantics."""
    out_features, in_features = validate_planar_w3(layout)
    codes = decode_planar_w3_codes(layout, original_order=False)
    groups = in_features // GROUP_SIZE
    signed = (codes.to(torch.int16) - ZERO).reshape(groups, GROUP_SIZE, out_features).permute(0, 2, 1)
    # The retained artifact was loaded with dtype=bfloat16, which casts scales
    # before GPTQModel's eager Torch dequantizer performs this multiplication.
    sorted_weight = (layout["scales"].unsqueeze(-1) * signed).permute(1, 0, 2).reshape(out_features, in_features)
    weight = torch.empty_like(sorted_weight)
    weight[:, layout["perm"].to(torch.int64)] = sorted_weight
    return weight.to(dtype=dtype).contiguous()


def structural_w3_matvec(x: torch.Tensor, layout: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Direct FP32 structural reference without materializing the whole weight."""
    out_features, in_features = validate_planar_w3(layout)
    if x.ndim != 2 or tuple(x.shape) != (1, in_features):
        raise ValueError("x must have shape [1,K]")
    if x.device != layout["planes"].device:
        raise ValueError("x and layout must share a device")
    sorted_codes = decode_planar_w3_codes(layout, original_order=False)
    sorted_x = x[0, layout["perm"].long()].float().reshape(-1, GROUP_SIZE)
    codes = (sorted_codes.to(torch.int16) - ZERO).reshape(-1, GROUP_SIZE, out_features)
    result = torch.zeros(out_features, dtype=torch.float32, device=x.device)
    for group in range(codes.shape[0]):
        dot = torch.matmul(codes[group].T.float(), sorted_x[group])
        result.add_(dot * layout["scales"][group].float())
    return result.unsqueeze(0)


def bf16_weight_semantics_w3_matvec(
    x: torch.Tensor,
    layout: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """FP32 grouped matvec with GPTQModel's per-weight BF16 rounding.

    A BF16 scale multiplied by signed W3 codes ``{-4,...,3}`` differs from
    ``scale * code`` only at codes ``+/-3``.  The explicit correction keeps
    the structural factorization and is a diagnostic oracle for the matching
    CUDA candidate; it does not claim native BF16 GEMM accumulation order.
    """
    out_features, in_features = validate_planar_w3(layout)
    if x.ndim != 2 or tuple(x.shape) != (1, in_features):
        raise ValueError("x must have shape [1,K]")
    if x.device != layout["planes"].device:
        raise ValueError("x and layout must share a device")
    sorted_codes = decode_planar_w3_codes(layout, original_order=False)
    sorted_x = x[0, layout["perm"].long()].float().reshape(-1, GROUP_SIZE)
    codes = (sorted_codes.to(torch.int16) - ZERO).reshape(-1, GROUP_SIZE, out_features)
    scales = layout["scales"]
    rounded_three = (scales * 3).to(torch.bfloat16).float()
    delta = rounded_three - 3.0 * scales.float()
    result = torch.zeros(out_features, dtype=torch.float32, device=x.device)
    for group in range(codes.shape[0]):
        signed = codes[group]
        activation = sorted_x[group].unsqueeze(-1)
        integer_dot = torch.sum(signed.float() * activation, dim=0)
        correction_dot = torch.sum(
            ((signed == 3).float() - (signed == -3).float()) * activation,
            dim=0,
        )
        result.add_(
            integer_dot * scales[group].float()
            + correction_dot * delta[group]
        )
    return result.unsqueeze(0)


def layout_storage(layout: Mapping[str, torch.Tensor]) -> dict[str, object]:
    out_features, in_features = validate_planar_w3(layout)
    sizes = {name: value.numel() * value.element_size() for name, value in layout.items()}
    total = sum(sizes.values())
    return {
        "format": FORMAT,
        "shape": [out_features, in_features],
        "tensor_bytes": sizes,
        "total_bytes": total,
        "bits_per_weight": total * 8 / (out_features * in_features),
    }


def pack_gptq_w3_codes(codes: torch.Tensor, *, packed_axis: int = 0) -> torch.Tensor:
    """Test/reference packer for the canonical GPTQ INT3 layout."""
    if codes.device.type != "cpu" or codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError("codes must be CPU uint8 rank two")
    logical = codes if packed_axis == 0 else codes.T
    if logical.shape[0] % 32 or bool((logical > 7).any()):
        raise ValueError("packed logical dimension must be divisible by 32 and codes must be W3")
    block = logical.to(torch.int64).reshape(-1, 32, logical.shape[1])
    shifts10 = (torch.arange(10, dtype=torch.int64) * 3).reshape(10, 1)
    a = torch.sum(block[:, :10] << shifts10, dim=1) | (block[:, 10] << 30)
    b = ((block[:, 10] >> 2) & 1) | torch.sum(
        block[:, 11:21] << (shifts10 + 1), dim=1
    ) | (block[:, 21] << 31)
    c = ((block[:, 21] >> 1) & 3) | torch.sum(
        block[:, 22:32] << (shifts10 + 2), dim=1
    )
    packed = torch.stack((a, b, c), dim=1).reshape(-1, logical.shape[1]).to(torch.int32)
    return packed if packed_axis == 0 else packed.T.contiguous()
