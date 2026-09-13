"""Fail-closed five-Linear admission gate for the 8B sequential runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import sha256_file
from .qwen3_8b import TARGETS, REVISION, linear_gate, validate_architecture, validate_linear_config


def relative_file(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("suite paths must be relative without parent traversal")
    return root / path


def validate_suite(suite: dict[str, Any], root: Path) -> None:
    expected = {f"model.layers.0.{m}.weight" for m in TARGETS}
    entries = suite.get("entries", [])
    if suite.get("status") != "passed" or len(entries) != 5:
        raise ValueError("five accepted Linear results are required")
    if {e["target"] for e in entries} != expected:
        raise ValueError("representative target coverage drifted")
    for entry in entries:
        files = {kind: relative_file(root, entry[kind]) for kind in ("result", "payload", "acceptance", "config")}
        for kind, path in files.items():
            if sha256_file(path) != entry[kind + "_sha256"]:
                raise ValueError(f"{kind} hash drifted: {entry['target']}")
        result = json.loads(files["result"].read_text())
        linear_config = json.loads(files["config"].read_text())
        validate_linear_config(linear_config)
        acceptance = json.loads(files["acceptance"].read_text())
        if acceptance.get("status") != "passed" or acceptance["target"] != entry["target"]:
            raise ValueError("Linear was not accepted")
        for kind in ("result", "payload", "config"):
            if acceptance[kind + "_sha256"] != entry[kind + "_sha256"]:
                raise ValueError("acceptance does not refer to this artifact")
        if result["model"]["target_tensor"] != entry["target"] or result["model"]["revision"] != REVISION:
            raise ValueError("Linear model drifted")
        if result["status"] != "completed_pending_review" or not result["decision"]["linear_gate_passed"]:
            raise ValueError("Linear did not pass")
        if result["config_sha256"] != entry["config_sha256"] or result["payload"]["sha256"] != entry["payload_sha256"]:
            raise ValueError("result provenance mismatch")
        if result["calibration"]["artifact_manifest_sha256"] != suite["calibration_manifest_sha256"]:
            raise ValueError("Linears used different calibration manifests")
        metrics = result["reconstruction"]
        gate = linear_gate(metrics["pure_two_base_obq"], metrics["hessian_salient_hybrid_s8_obq"], metrics["hybrid_maximum_delta_outside_selected_columns"])
        if not gate["linear_gate_passed"]:
            raise ValueError("Linear error/support gate failed")


def validate_full_contract(config: dict[str, Any], args: Any) -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads((root / "configs/experiments/qwen3_8b_full_hessian_obq_s8_v1.json").read_text())
    for key in ("schema_version", "seed", "model", "algorithm", "layerwise_calibration", "global_solver", "refinement_solver", "artifact_policy", "execution", "decision"):
        if config[key] != baseline[key]:
            raise ValueError(f"8B full-model contract drifted: {key}")
    if not config.get("accepted_linear_suite_sha256"):
        raise ValueError("resolve the full config only after five Linear acceptances")
    if sha256_file(args.linear_suite) != config["accepted_linear_suite_sha256"]:
        raise ValueError("Linear suite hash drifted")
    suite = json.loads(args.linear_suite.read_text())
    validate_suite(suite, args.project_root)
    if suite["calibration_manifest_sha256"] != config["accepted_calibration"]["manifest_sha256"]:
        raise ValueError("suite calibration differs from full model")
    # Require exactly the complete shard/metadata set from the actual index.
    index = json.loads((args.snapshot_root / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values()) | {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    if set(config["model_preflight_files"]) != expected:
        raise ValueError("incomplete model preflight hash set")
    for name, digest in config["model_preflight_files"].items():
        if sha256_file(relative_file(args.snapshot_root, name)) != digest:
            raise ValueError(f"model snapshot drifted: {name}")
    validate_architecture(json.loads((args.snapshot_root / "config.json").read_text()))
    if args.artifact_dir.name != args.arm or args.artifact_dir.parent.name != config["artifact_policy"]["artifact_id"]:
        raise ValueError("use the versioned 8B artifact directory and separate arm subdirectories")
