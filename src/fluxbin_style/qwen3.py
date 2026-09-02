"""Fail-closed Qwen3 transformer-block Linear inventory."""

from __future__ import annotations

import re
from typing import Any, Iterable


_QWEN3_LINEAR = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\."
    r"(?P<module>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


QWEN3_LINEAR_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def qwen3_linear_weight_names(num_layers: int) -> tuple[str, ...]:
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    return tuple(
        f"model.layers.{layer}.{module}.weight"
        for layer in range(num_layers)
        for module in QWEN3_LINEAR_MODULES
    )


def parse_qwen3_linear_name(name: str) -> tuple[int, str]:
    match = _QWEN3_LINEAR.fullmatch(name)
    if match is None:
        raise ValueError(f"not a Qwen3 transformer-block Linear: {name}")
    return int(match.group("layer")), match.group("module")


def expected_qwen3_linear_shape(config: dict[str, Any], module: str) -> list[int]:
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    values: dict[str, int] = {}
    for key in required:
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Qwen3 config requires positive integer {key}")
        values[key] = value
    hidden = values["hidden_size"]
    if module == "self_attn.q_proj":
        return [values["num_attention_heads"] * values["head_dim"], hidden]
    if module in ("self_attn.k_proj", "self_attn.v_proj"):
        return [values["num_key_value_heads"] * values["head_dim"], hidden]
    if module == "self_attn.o_proj":
        return [hidden, values["num_attention_heads"] * values["head_dim"]]
    if module in ("mlp.gate_proj", "mlp.up_proj"):
        return [values["intermediate_size"], hidden]
    if module == "mlp.down_proj":
        return [hidden, values["intermediate_size"]]
    raise ValueError(f"unsupported Qwen3 module: {module}")


def build_qwen3_linear_inventory(
    config: dict[str, Any],
    tensors: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate complete seven-Linear-per-layer coverage and all rank-two weights."""

    if config.get("model_type") != "qwen3":
        raise ValueError("expected model_type='qwen3'")
    num_layers = config.get("num_hidden_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("invalid num_hidden_layers")
    expected = set(qwen3_linear_weight_names(num_layers))
    seen: set[str] = set()
    included: set[str] = set()
    records: list[dict[str, Any]] = []
    for tensor in tensors:
        name = tensor.get("name")
        shape = tensor.get("shape")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"invalid or duplicate tensor name: {name!r}")
        seen.add(name)
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension <= 0 for dimension in shape
        ):
            raise ValueError(f"invalid shape for {name}")
        match = _QWEN3_LINEAR.fullmatch(name)
        if match is not None:
            layer = int(match.group("layer"))
            module = match.group("module")
            if layer >= num_layers:
                raise ValueError(f"tensor exceeds configured layer count: {name}")
            if shape != expected_qwen3_linear_shape(config, module):
                raise ValueError(f"shape drifted for {name}: {shape}")
            included.add(name)
            records.append({**tensor, "decision": "include"})
        elif len(shape) == 2 and name not in ("model.embed_tokens.weight", "lm_head.weight"):
            raise ValueError(f"unclassified rank-two tensor: {name}")
        else:
            records.append({**tensor, "decision": "exclude"})
    if included != expected:
        raise ValueError(
            f"Qwen3 Linear coverage mismatch: missing={sorted(expected-included)}, "
            f"unexpected={sorted(included-expected)}"
        )
    included_records = [record for record in records if record["decision"] == "include"]
    return {
        "status": "passed",
        "included_tensor_count": len(included_records),
        "included_parameter_count": sum(
            _numel(record["shape"]) for record in included_records
        ),
        "excluded_tensor_count": len(records) - len(included_records),
        "tensors": records,
    }


def _numel(shape: list[int]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value
