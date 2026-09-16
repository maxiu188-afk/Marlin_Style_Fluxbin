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
