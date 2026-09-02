#!/usr/bin/env python3
"""Materialize deterministic C4 calibration token sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

from fluxbin_style import atomic_json, sha256_file, tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any], tokenizer_root: Path) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if config["artifact_id"] != "qwen3-32b-c4-calibration-256x2048-v1":
        raise ValueError("calibration artifact id drifted")
    dataset = config["dataset"]
    if dataset != {
        "repo_id": "allenai/c4",
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "configuration": "en",
        "split": "train",
        "data_files": ["en/c4-train.00000-of-01024.json.gz"],
    }:
        raise ValueError("C4 source contract drifted")
    if config["sampling"] != {
        "method": "random_document_then_random_contiguous_span_with_replacement",
        "sequences": 256,
        "sequence_length": 2048,
        "minimum_document_tokens": 2049,
        "special_tokens": False,
    }:
        raise ValueError("calibration sampling contract drifted")
    tokenizer = config["tokenizer"]
    if tokenizer["revision"] != tokenizer_root.name:
        raise ValueError("tokenizer snapshot does not match pinned revision")
    if tokenizer["repo_id"] != "Qwen/Qwen3-32B" or not tokenizer["use_fast"]:
        raise ValueError("tokenizer contract drifted")


def implementation_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.tokens, args.output, args.source_manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config, args.tokenizer_root)
    args.tokens.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_manifest.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["repo_id"],
        dataset_config["configuration"],
        split=dataset_config["split"],
        data_files=dataset_config["data_files"],
        revision=dataset_config["revision"],
    )
    if len(dataset) <= 0:
        raise RuntimeError("C4 dataset shard is empty")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root,
        local_files_only=True,
        use_fast=True,
    )
    sampling = config["sampling"]
    sequence_length = sampling["sequence_length"]
    random_source = random.Random(config["seed"])
    sequences: list[torch.Tensor] = []
    selections: list[dict[str, int]] = []
    rejected_short_documents = 0
    while len(sequences) < sampling["sequences"]:
        document_index = random_source.randrange(len(dataset))
        encoded = tokenizer(
            dataset[document_index]["text"],
            return_tensors="pt",
            add_special_tokens=sampling["special_tokens"],
        ).input_ids[0]
        if encoded.numel() < sampling["minimum_document_tokens"]:
            rejected_short_documents += 1
            continue
        start = random_source.randrange(encoded.numel() - sequence_length)
        stop = start + sequence_length
        sequence = encoded[start:stop]
        if sequence.numel() != sequence_length:
            raise RuntimeError("sampled sequence length drifted")
        sequences.append(sequence.to(torch.int32))
        selections.append(
            {
                "sequence_index": len(sequences) - 1,
                "document_index": document_index,
                "document_token_count": int(encoded.numel()),
                "start_token": start,
                "stop_token": stop,
            }
        )

    token_ids = torch.stack(sequences).contiguous()
    if tuple(token_ids.shape) != (256, 2048):
        raise RuntimeError("calibration token shape drifted")
    if torch.any(token_ids < 0):
        raise RuntimeError("token artifact contains negative ids")
    temporary_tokens = args.tokens.with_name(f".{args.tokens.name}.tmp-{os.getpid()}")
    save_file({"token_ids": token_ids}, temporary_tokens)
    os.replace(temporary_tokens, args.tokens)

    source_files = [args.config, Path(__file__)]
    args.source_manifest.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in source_files),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "artifact_id": config["artifact_id"],
        "config": config,
        "dataset": {
            "rows_in_selected_shard": len(dataset),
            "fingerprint": dataset._fingerprint,
            "rejected_short_documents": rejected_short_documents,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "name_or_path": tokenizer.name_or_path,
        },
        "tokens": {
            "path": args.tokens.name,
            "shape": list(token_ids.shape),
            "dtype": str(token_ids.dtype),
            "minimum_id": int(token_ids.min()),
            "maximum_id": int(token_ids.max()),
            "tensor_sha256": tensor_sha256(token_ids),
            "file_sha256": sha256_file(args.tokens),
            "bytes": args.tokens.stat().st_size,
        },
        "selections": selections,
        "source_manifest": {
            "path": args.source_manifest.name,
            "sha256": sha256_file(args.source_manifest),
        },
        "implementation_sha256": implementation_sha256(source_files),
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_C4_TOKENS={args.tokens}")
    print(f"FLUXBIN_C4_RESULT={args.output}")


if __name__ == "__main__":
    main()
