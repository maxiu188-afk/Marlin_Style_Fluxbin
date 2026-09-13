"""Offline Qwen3-8B preflight and frozen layer-zero Linear contract.

No downloads or CUDA execution happen in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .evaluation import sha256_file, tensor_sha256
from .qwen3 import build_qwen3_linear_inventory


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ARCHITECTURE = {
    "model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
    "intermediate_size": 12288, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128,
}
TARGETS = {
    "self_attn.o_proj": [4096, 4096],
    "self_attn.q_proj": [4096, 4096],
    "self_attn.k_proj": [1024, 4096],
    "mlp.gate_proj": [12288, 4096],
    "mlp.down_proj": [4096, 12288],
}


def validate_architecture(config: dict[str, Any]) -> None:
    for key, expected in ARCHITECTURE.items():
        if config.get(key) != expected:
            raise ValueError(f"8B architecture drift: {key}")


def snapshot_preflight(root: Path) -> dict[str, Any]:
    """Hash local shards and inspect headers, loading only five target tensors.

    Snapshot directory name is a revision assertion, not remote authenticity
    verification. The recorded content hashes must be retained for review.
    """
    if root.name != REVISION:
        raise ValueError("expected pinned 8B snapshot directory")
    config = json.loads((root / "config.json").read_text())
    validate_architecture(config)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    records, hashes, targets = [], {}, {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError("invalid shard path")
        shard = root / filename
        hashes[filename] = sha256_file(shard)
        with safe_open(shard, framework="pt", device="cpu") as f:
            expected_names = {n for n, s in weight_map.items() if s == filename}
            if set(f.keys()) != expected_names:
                raise ValueError("shard/index inventory mismatch")
            for name in f.keys():
                view = f.get_slice(name)
                records.append({"name": name, "shape": view.get_shape(), "dtype": view.get_dtype()})
                if name.startswith("model.layers.") and name.endswith(".weight") and len(view.get_shape()) == 2:
                    if view.get_dtype() != "BF16":
                        raise ValueError(f"expected BF16: {name}")
                if name in {f"model.layers.0.{m}.weight" for m in TARGETS}:
                    targets[name] = tensor_sha256(f.get_tensor(name))
    inventory = build_qwen3_linear_inventory(config, records)
    if inventory["included_tensor_count"] != 252 or inventory["included_parameter_count"] != 6945767424:
        raise ValueError("8B included weight inventory drifted")
    for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        hashes[filename] = sha256_file(root / filename)
    return {"status": "passed", "revision": REVISION, "files": hashes,
            "inventory": inventory, "target_tensor_sha256": targets}


def validate_linear_config(config: dict[str, Any]) -> None:
    model = config["model"]
    if model["repo_id"] != "Qwen/Qwen3-8B" or model["revision"] != REVISION:
        raise ValueError("expected pinned Qwen3-8B")
    candidates = {f"model.layers.0.{m}.weight": shape for m, shape in TARGETS.items()}
    if candidates.get(model["target_tensor"]) != model["target_shape"]:
        raise ValueError("only declared layer-0 representative targets are supported")
    if config["seed"] != 20260902:
        raise ValueError("seed drifted")
    if config.get("schema_version") != 2 or model["checkpoint_dtype"] != "torch.bfloat16":
        raise ValueError("schema or dtype drifted")
    # Compare the frozen semantic fields to the versioned baseline, not hardware.
    baseline_path = Path(__file__).resolve().parents[2] / "configs/experiments/qwen3_8b_single_linear_hessian_obq_s8_v1.json"
    baseline = json.loads(baseline_path.read_text())
    for key in ("algorithm", "global_solver", "refinement_solver", "calibration", "decision", "execution"):
        if config[key] != baseline[key]:
            raise ValueError(f"frozen contract drifted: {key}")
    preflight = config.get("preflight")
    if not preflight or preflight.get("status") != "passed":
        raise ValueError("run offline snapshot preflight first")
    if preflight.get("revision") != REVISION:
        raise ValueError("preflight revision drifted")
    for name in preflight["files"]:
        if Path(name).name != name:
            raise ValueError("invalid preflight file path")
    if not {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}.issubset(preflight["files"]):
        raise ValueError("missing snapshot metadata hashes")
    if not model.get("target_tensor_sha256") or preflight["target_tensor_sha256"].get(model["target_tensor"]) != model["target_tensor_sha256"]:
        raise ValueError("target is not hash-pinned by preflight")


def linear_gate(pure: dict[str, float], hybrid: dict[str, float], outside: float) -> dict[str, Any]:
    """Require both weight and same-input calibration error improvement."""
    import math
    finite = all(math.isfinite(x) for m in (pure, hybrid) for x in m.values())
    weight = hybrid["squared_error"] < pure["squared_error"]
    output = hybrid["calibration_total_output_squared_error"] < pure["calibration_total_output_squared_error"]
    passed = finite and weight and output and outside == 0
    return {"linear_gate_passed": passed, "metrics_finite": finite,
            "hybrid_weight_sse_improved": weight, "hybrid_calibration_loss_improved": output,
            "exact_sparse_support": outside == 0, "manual_review_required": True,
            "full_model_auto_launch": False, "ppl_auto_launch": False, "backend_auto_launch": False}
