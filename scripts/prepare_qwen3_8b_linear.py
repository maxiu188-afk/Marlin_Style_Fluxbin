#!/usr/bin/env python3
"""Resolve one 8B Linear config against a local pinned snapshot; no CUDA/download."""
import argparse
import json
from pathlib import Path

from fluxbin_style import atomic_json
from fluxbin_style.qwen3_8b import TARGETS, snapshot_preflight, validate_linear_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/experiments/qwen3_8b_single_linear_hessian_obq_s8_v1.json").read_text())
    preflight = snapshot_preflight(args.snapshot_root)
    name = f"model.layers.0.{args.target}.weight"
    config["model"].update(target_tensor=name, target_shape=TARGETS[args.target],
                           target_tensor_sha256=preflight["target_tensor_sha256"][name])
    config["preflight"] = preflight
    validate_linear_config(config)
    atomic_json(args.output, config)
    print(f"Prepared {name}: {args.output}; no model execution launched")


if __name__ == "__main__":
    main()
