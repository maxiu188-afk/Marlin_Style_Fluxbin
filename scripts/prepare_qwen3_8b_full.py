#!/usr/bin/env python3
"""Bind a full-model config to five accepted 8B Linears; never launch a run."""
import argparse
import json
from pathlib import Path

from fluxbin_style import atomic_json, sha256_file
from fluxbin_style.qwen3_8b import TARGETS
from fluxbin_style.qwen3_8b_full import validate_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-output", type=Path, required=True)
    args = parser.parse_args()
    for p in (args.output, args.suite_output):
        if p.exists():
            raise FileExistsError(p)
    root = args.project_root.resolve()
    c = json.loads((root / "configs/experiments/qwen3_8b_full_hessian_obq_s8_v1.json").read_text())
    cal = root / "artifacts/qwen3-8b-c4-v1/manifest.json"
    manifest = json.loads(cal.read_text())
    entries = []
    for module in TARGETS:
        stem = module.split(".")[-1]
        entry = {"target": f"model.layers.0.{module}.weight",
                 "result": f"results/qwen3-8b-linear-v1/{stem}.json",
                 "acceptance": f"results/qwen3-8b-linear-v1/{stem}.acceptance.json",
                 "config": f"results/qwen3-8b-linear-v1/{stem}.config.json",
                 "payload": f"artifacts/qwen3-8b-linear-v1/{stem}.safetensors"}
        for kind in ("result", "acceptance", "config", "payload"):
            entry[kind + "_sha256"] = sha256_file(root / entry[kind])
        entries.append(entry)
    suite = {"status": "passed", "entries": entries, "calibration_manifest_sha256": sha256_file(cal)}
    validate_suite(suite, root)
    o = json.loads((root / entries[0]["result"]).read_text())
    oc = json.loads((root / entries[0]["config"]).read_text())
    c["model_preflight_files"] = oc["preflight"]["files"]
    c["accepted_calibration"] = {
        "artifact_id": manifest["artifact_id"], "manifest_sha256": sha256_file(cal),
        "token_file_sha256": manifest["tokens"]["file_sha256"],
        "token_tensor_sha256": manifest["tokens"]["tensor_sha256"],
        "dataset_revision": manifest["config"]["dataset"]["revision"],
        "sequences": 256, "sequence_length": 2048,
    }
    c["accepted_single_linear_gate"] = {
        "outcome": "go_full_model", "result_sha256": entries[0]["result_sha256"],
        "payload_sha256": entries[0]["payload_sha256"],
        "pure_calibration_loss": o["reconstruction"]["pure_two_base_obq"]["calibration_total_output_squared_error"],
        "hybrid_calibration_loss": o["reconstruction"]["hessian_salient_hybrid_s8_obq"]["calibration_total_output_squared_error"],
    }
    atomic_json(args.suite_output, suite)
    c["accepted_linear_suite_sha256"] = sha256_file(args.suite_output)
    atomic_json(args.output, c)
    print(f"Prepared {args.output}; full-model execution not launched")


if __name__ == "__main__":
    main()
