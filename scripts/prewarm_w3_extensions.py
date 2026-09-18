#!/usr/bin/env python3
"""Prewarm content-addressed W3 CUDA extensions and record an exact manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path

import torch

from fluxbin_style.evaluation import atomic_json, sha256_file
from fluxbin_style.w3_lut_deployment import (
    EXPERIMENTAL_ROW_TILES,
    load_w3_lut_extension,
    w3_lut_extension_build,
)


def parse_tiles(value: str) -> list[int]:
    try:
        tiles = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("row tiles must be comma-separated integers") from exc
    if not tiles or len(set(tiles)) != len(tiles):
        raise argparse.ArgumentTypeError("row tiles must be nonempty and unique")
    if any(tile not in EXPERIMENTAL_ROW_TILES for tile in tiles):
        raise argparse.ArgumentTypeError(f"row tiles must be chosen from {EXPERIMENTAL_ROW_TILES}")
    return tiles


def shared_objects(build_directory: Path) -> list[dict]:
    records = [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(build_directory.glob("*.so"))
    ]
    if not records:
        raise RuntimeError(f"no shared object produced in {build_directory}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-tiles", type=parse_tiles, default=parse_tiles("256,512,1024"))
    parser.add_argument("--include-uniform-probes", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("prewarm requires NVIDIA CUDA")
    if not os.environ.get("FLUXBIN_EXTENSION_CACHE_ROOT"):
        raise RuntimeError("set FLUXBIN_EXTENSION_CACHE_ROOT to a persistent directory")

    variants = [(tile, False) for tile in args.row_tiles]
    if args.include_uniform_probes:
        variants += [(tile, True) for tile in args.row_tiles]
    records = []
    started = time.monotonic()
    for row_tile, uniform_probe in variants:
        kwargs, contract = w3_lut_extension_build(row_tile, uniform_probe)
        build_directory = kwargs.get("build_directory")
        if not build_directory:
            raise RuntimeError("content-addressed build directory was not selected")
        variant_started = time.monotonic()
        load_w3_lut_extension(row_tile, uniform_probe)
        records.append(
            {
                "row_tile": row_tile,
                "uniform_index_probe": uniform_probe,
                "contract": contract,
                "build_directory": build_directory,
                "shared_objects": shared_objects(Path(build_directory)),
                "elapsed_seconds": time.monotonic() - variant_started,
            }
        )
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "status": "prewarmed",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cache_root": str(Path(os.environ["FLUXBIN_EXTENSION_CACHE_ROOT"]).resolve()),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "variants": records,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"FLUXBIN_W3_PREWARMED={len(records)}")


if __name__ == "__main__":
    main()
