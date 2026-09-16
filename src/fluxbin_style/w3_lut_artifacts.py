"""Hash-bound access to offline W3 LUT deployment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file

from .evaluation import sha256_file
from .gptq_deployment import FIELDS, FORMAT, validate_planar_w3
from .qwen3 import QWEN3_LINEAR_MODULES


def load_w3_lut_layer(
    root: Path, layer: int, *, expected_manifest_sha256: str
) -> tuple[dict, dict]:
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("W3 LUT manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_pending_gpu_validation" or manifest.get("format") != FORMAT:
        raise ValueError("W3 LUT manifest status/format drift")
    semantics = manifest.get("canonical_semantics_gate", {})
    if (
        semantics.get("status") != "confirmed_before_payload_writes"
        or semantics.get("inspected_linears") != 252
        or semantics.get("all_groups_have_128_columns") is not True
        or semantics.get("decoded_zero_unique") != [4]
    ):
        raise ValueError("W3 LUT canonical-semantics gate missing or drifted")
    if not 0 <= layer < 36:
        raise ValueError("layer outside 0..35")
    relative = f"payloads/layer-{layer:03d}.safetensors"
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    path = root / relative
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError("W3 LUT layer hash/size drift")
    tensors = load_file(path)
    expected = {f"{module}.{field}" for module in QWEN3_LINEAR_MODULES for field in FIELDS}
    if set(tensors) != expected:
        raise ValueError("W3 LUT layer inventory drift")
    for module in QWEN3_LINEAR_MODULES:
        validate_planar_w3({field: tensors[f"{module}.{field}"] for field in FIELDS})
    return tensors, entry


def replace_w3_model_linears(
    model,
    artifact_root: Path,
    *,
    expected_manifest_sha256: str,
    allow_prefill_fallback: bool = False,
) -> dict:
    """Install all 252 hash-bound W3 Linears without changing attention or cache code."""
    from torch import nn

    from .qwen3 import expected_qwen3_linear_shape
    from .qwen3_8b import ARCHITECTURE, validate_architecture
    from .w3_lut_deployment import QWEN3_ROW_TILE_BY_SHAPE, replace_w3_block_linears

    validate_architecture(model.config.to_dict())
    if len(model.model.layers) != 36:
        raise ValueError("expected 36 decoder layers")
    records = []
    for layer in range(36):
        payload, entry = load_w3_lut_layer(
            artifact_root, layer, expected_manifest_sha256=expected_manifest_sha256
        )
        for name in QWEN3_LINEAR_MODULES:
            shape = validate_planar_w3({field: payload[f"{name}.{field}"] for field in FIELDS})
            if list(shape) != expected_qwen3_linear_shape(ARCHITECTURE, name):
                raise ValueError(f"8B W3 shape drift: layer {layer} {name}")
            target = model.model.layers[layer].get_submodule(name)
            if not isinstance(target, nn.Linear) or (
                target.out_features,
                target.in_features,
            ) != shape:
                raise ValueError(f"Linear shape/type mismatch: layer {layer} {name}")
        records.append(entry)
    del payload

    coverage = []
    for layer, block in enumerate(model.model.layers):
        payload, _ = load_w3_lut_layer(
            artifact_root, layer, expected_manifest_sha256=expected_manifest_sha256
        )
        modules = replace_w3_block_linears(
            block,
            payload,
            fallback="dense" if allow_prefill_fallback else "error",
        )
        coverage.extend(f"model.layers.{layer}.{name}" for name in modules)
    return {
        "format": FORMAT,
        "manifest_sha256": expected_manifest_sha256,
        "layers": records,
        "coverage": coverage,
        "linear_count": len(coverage),
        "row_tile_by_shape": {
            f"{out_features}x{in_features}": row_tile
            for (out_features, in_features), row_tile in QWEN3_ROW_TILE_BY_SHAPE.items()
        },
        "prefill_fallback_enabled": allow_prefill_fallback,
        "status": "installed_not_validated",
    }
