#!/usr/bin/env python3
"""Quantize Qwen3-8B to symmetric GPTQ W3 g128 and decode 252 BF16 weights.

GPTQModel is used only to create and reload the packed artifact.  The output is
a compact per-layer dense-BF16 payload consumed by the repository's frozen PPL
scorer; no GPTQ inference kernel participates in quality evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import QWEN3_LINEAR_MODULES, atomic_json, sha256_file, tensor_sha256
from fluxbin_style.qwen3 import expected_qwen3_linear_shape, qwen3_linear_weight_names
from fluxbin_style.qwen3_8b import validate_architecture
from fluxbin_style.rate_distortion import (
    EXPECTED_LAYERS,
    EXPECTED_LINEARS,
    EXPECTED_WEIGHTS,
    analytical_gptq_storage,
    classify_gptq_tensor,
    tensor_nbytes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_static_config(config: dict[str, Any]) -> None:
    model = config["model"]
    gptq = config["gptq"]
    calibration = config["calibration"]
    execution = config["execution"]
    expected = {
        "schema_version": 1,
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "layers": EXPECTED_LAYERS,
        "linears": EXPECTED_LINEARS,
        "weights": EXPECTED_WEIGHTS,
        "bits": 3,
        "group_size": 128,
        "sym": True,
        "desc_act": True,
        "act_group_aware": False,
        "static_groups": False,
        "true_sequential": True,
        "lm_head": False,
        "mse": 0.0,
        "damp_percent": 0.01,
        "damp_auto_increment": 0.01,
        "fallback": None,
        "format": "gptq",
        "pack_dtype": "torch.int32",
        "pack_impl": "cpu",
        "quantization_backend": "gptq_torch",
        "decode_backend": "gptq_torch",
        "quantize_embedding": False,
        "expected_quantized_modules": EXPECTED_LINEARS,
        "post_save_reload_required": True,
        "decoded_dtype": "torch.bfloat16",
        "samples": 256,
        "sequence_length": 2048,
        "batch_size": 1,
        "reuse_exact_token_ids": True,
        "sorting": None,
        "concatenation": None,
        "quality_device_names": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
        "compute_capability": [8, 0],
        "torch": "2.8.0+cu128",
        "transformers": "5.14.1",
        "datasets": "5.0.0",
        "safetensors": "0.8.0",
    }
    observed = {
        "schema_version": config.get("schema_version"),
        "model_revision": model.get("revision"),
        "layers": model.get("expected_hidden_layers"),
        "linears": model.get("expected_tensor_count"),
        "weights": model.get("expected_parameter_count"),
        "bits": gptq.get("bits"),
        "group_size": gptq.get("group_size"),
        "sym": gptq.get("sym"),
        "desc_act": gptq.get("desc_act"),
        "act_group_aware": gptq.get("act_group_aware"),
        "static_groups": gptq.get("static_groups"),
        "true_sequential": gptq.get("true_sequential"),
        "lm_head": gptq.get("lm_head"),
        "mse": gptq.get("mse"),
        "damp_percent": gptq.get("damp_percent"),
        "damp_auto_increment": gptq.get("damp_auto_increment"),
        "fallback": gptq.get("fallback"),
        "format": gptq.get("format"),
        "pack_dtype": gptq.get("pack_dtype"),
        "pack_impl": gptq.get("pack_impl"),
        "quantization_backend": gptq.get("quantization_backend"),
        "decode_backend": gptq.get("decode_backend"),
        "quantize_embedding": gptq.get("quantize_embedding"),
        "expected_quantized_modules": gptq.get("expected_quantized_modules"),
        "post_save_reload_required": gptq.get("post_save_reload_required"),
        "decoded_dtype": gptq.get("decoded_dtype"),
        "samples": calibration.get("sample_count"),
        "sequence_length": calibration.get("sequence_length"),
        "batch_size": calibration.get("batch_size"),
        "reuse_exact_token_ids": calibration.get("reuse_exact_token_ids"),
        "sorting": calibration.get("sorting"),
        "concatenation": calibration.get("concatenation"),
        "quality_device_names": execution.get("quality_device_names"),
        "compute_capability": execution.get("compute_capability"),
        "torch": execution.get("torch"),
        "transformers": execution.get("transformers"),
        "datasets": execution.get("datasets"),
        "safetensors": execution.get("safetensors"),
    }
    if observed != expected:
        raise ValueError(f"frozen W3 experiment config drifted: {observed}")


def validate_inputs(config: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    validate_static_config(config)
    model = config["model"]
    if args.snapshot_root.name != model["revision"]:
        raise ValueError("snapshot directory does not match pinned model revision")
    architecture = json.loads((args.snapshot_root / "config.json").read_text(encoding="utf-8"))
    validate_architecture(architecture)
    for name, digest in model["model_preflight_files"].items():
        if sha256_file(args.snapshot_root / name) != digest:
            raise ValueError(f"model snapshot file drifted: {name}")
    calibration = config["calibration"]
    if sha256_file(args.calibration_manifest) != calibration["manifest_sha256"]:
        raise ValueError("calibration manifest drifted")
    if sha256_file(args.calibration_tokens) != calibration["token_file_sha256"]:
        raise ValueError("calibration token artifact drifted")
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed" or manifest.get("artifact_id") != calibration["artifact_id"]:
        raise ValueError("calibration artifact is not the accepted Qwen3-8B artifact")
    if manifest["config"]["dataset"]["revision"] != calibration["revision"]:
        raise ValueError("calibration dataset revision drifted")
    with safe_open(args.calibration_tokens, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"token_ids"}:
            raise ValueError("calibration token inventory drifted")
        tokens = handle.get_tensor("token_ids")
    if list(tokens.shape) != [calibration["sample_count"], calibration["sequence_length"]]:
        raise ValueError("calibration token shape drifted")
    if tensor_sha256(tokens) != calibration["token_tensor_sha256"]:
        raise ValueError("calibration token tensor drifted")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("calibration tokens must be integer IDs")
    return tokens.to(torch.int64)


def validate_runtime(config: dict[str, Any]) -> None:
    expected = config["execution"]
    if not torch.cuda.is_available():
        raise RuntimeError("the formal W3 quantization requires NVIDIA CUDA")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) not in expected["quality_device_names"]:
        raise RuntimeError("unexpected quality-experiment GPU")
    if list(torch.cuda.get_device_capability(0)) != expected["compute_capability"]:
        raise RuntimeError("unexpected compute capability")
    observed = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "safetensors": __import__("safetensors").__version__,
    }
    for name, version in observed.items():
        if version != expected[name]:
            raise RuntimeError(f"{name} runtime drifted: {version}")
    gptq_version = importlib.metadata.version("gptqmodel")
    if gptq_version != config["gptq"]["version"]:
        raise RuntimeError(f"gptqmodel runtime drifted: {gptq_version}")


def target_modules() -> tuple[str, ...]:
    return tuple(name.removesuffix(".weight") for name in qwen3_linear_weight_names(EXPECTED_LAYERS))


def iter_quantized_safetensors(root: Path):
    for path in sorted(root.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if classify_gptq_tensor(name) is not None:
                    yield path, name, handle.get_tensor(name)


def audit_raw_artifact(raw_dir: Path) -> dict[str, Any]:
    targets = set(target_modules())
    fields: dict[str, set[str]] = {name: set() for name in targets}
    categories: dict[str, int] = {}
    category_dtypes: dict[str, set[str]] = {}
    unexpected_quantized: list[str] = []
    for _path, tensor_name, value in iter_quantized_safetensors(raw_dir):
        category = classify_gptq_tensor(tensor_name)
        assert category is not None
        module_name, suffix = tensor_name.rsplit(".", 1)
        if module_name not in targets:
            unexpected_quantized.append(tensor_name)
            continue
        if suffix in {"qweight", "qzeros", "g_idx"} and value.dtype != torch.int32:
            raise ValueError(f"GPTQ packed integer dtype drifted: {tensor_name}={value.dtype}")
        if suffix == "scales" and (
            value.dtype not in {torch.float16, torch.bfloat16} or value.element_size() != 2
        ):
            raise ValueError(f"GPTQ scale dtype drifted: {tensor_name}={value.dtype}")
        fields[module_name].add(suffix)
        categories[category] = categories.get(category, 0) + tensor_nbytes(value)
        category_dtypes.setdefault(category, set()).add(str(value.dtype))
    required = {"qweight", "scales", "qzeros", "g_idx"}
    invalid = {name: sorted(value) for name, value in fields.items() if value != required}
    if invalid or unexpected_quantized:
        raise ValueError(
            f"GPTQ packed coverage drifted: invalid={invalid}, unexpected={unexpected_quantized}"
        )
    weight_files = sorted(raw_dir.rglob("*.safetensors"))
    if not weight_files:
        raise ValueError("GPTQModel saved no safetensors artifact")
    tensor_bytes = sum(categories.values())
    return {
        "quantized_module_count": len(fields),
        "quantized_modules": sorted(fields),
        "tensor_categories_bytes": dict(sorted(categories.items())),
        "tensor_category_dtypes": {
            name: sorted(values) for name, values in sorted(category_dtypes.items())
        },
        "quantized_tensor_bytes": tensor_bytes,
        # GPTQModel saves a loadable whole-model checkpoint.  These files also
        # contain pass-through embeddings, lm_head and normalization tensors,
        # so their container size is provenance only, not the W3 rate.
        "serialized_model_artifact_file_bytes": sum(path.stat().st_size for path in weight_files),
        "weight_files": [
            {
                "path": str(path.relative_to(raw_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_files
        ],
    }


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def force_eager_torch_dequantizer() -> None:
    os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "0"


@torch.no_grad()
def decode_artifact(config: dict[str, Any], raw_dir: Path, decoded_dir: Path) -> list[dict[str, Any]]:
    # TorchLinear otherwise enables its optional Triton dequantizer when Triton
    # happens to be installed.  The quality artifact must use the audited eager
    # Torch 3-bit unpack path regardless of the server image.
    force_eager_torch_dequantizer()
    from gptqmodel import BACKEND, GPTQModel

    if decoded_dir.exists():
        raise FileExistsError(decoded_dir)
    decoded_dir.mkdir(parents=True)
    loaded = GPTQModel.load(
        str(raw_dir),
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = loaded.model
    architecture = json.loads((raw_dir / "config.json").read_text(encoding="utf-8"))
    validate_architecture(architecture)
    records = []
    for layer_index in range(EXPECTED_LAYERS):
        tensors: dict[str, torch.Tensor] = {}
        tensor_records = []
        for module_name in QWEN3_LINEAR_MODULES:
            full_name = f"model.layers.{layer_index}.{module_name}"
            module = model.get_submodule(full_name)
            if not all(hasattr(module, name) for name in ("qweight", "qzeros", "scales", "g_idx", "dequantize_weight")):
                raise TypeError(f"target is not a reloadable GPTQ Torch module: {full_name}")
            if getattr(module, "_triton_dequant_enabled", None) is not False:
                raise RuntimeError(f"Triton dequantization was not disabled: {full_name}")
            if (
                int(module.bits) != config["gptq"]["bits"]
                or int(module.group_size) != config["gptq"]["group_size"]
                or bool(module.sym) is not config["gptq"]["sym"]
                or bool(module.desc_act) is not config["gptq"]["desc_act"]
            ):
                raise ValueError(f"GPTQ module config drifted: {full_name}")
            weight = module.dequantize_weight().T.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            expected_shape = expected_qwen3_linear_shape(architecture, module_name)
            if list(weight.shape) != expected_shape or not torch.isfinite(weight).all():
                raise ValueError(f"decoded GPTQ weight is invalid: {full_name}")
            key = f"{module_name}.weight"
            tensors[key] = weight
            tensor_records.append(
                {
                    "module": module_name,
                    "shape": expected_shape,
                    "dtype": str(weight.dtype),
                    "numel": weight.numel(),
                    "sha256": tensor_sha256(weight),
                }
            )
        path = decoded_dir / f"layer-{layer_index:03d}.safetensors"
        save_file(tensors, path)
        with safe_open(path, framework="pt", device="cpu") as reopened:
            if set(reopened.keys()) != set(tensors):
                raise RuntimeError(f"decoded layer inventory changed after save: {layer_index}")
            for name, expected in ((item["module"] + ".weight", item["sha256"]) for item in tensor_records):
                if tensor_sha256(reopened.get_tensor(name)) != expected:
                    raise RuntimeError(f"decoded layer tensor changed after save: {layer_index}:{name}")
        records.append(
            {
                "layer": layer_index,
                "path": str(path.relative_to(decoded_dir.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensors": tensor_records,
            }
        )
        print(f"FLUXBIN_GPTQ_DECODED_LAYER={layer_index + 1}/{EXPECTED_LAYERS}", flush=True)
    del loaded, model
    gc.collect()
    return records


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokens = validate_inputs(config, args)
    if args.validate_only:
        print("FLUXBIN_GPTQ_W3_PREFLIGHT=passed; no quantization launched")
        return
    validate_runtime(config)
    force_eager_torch_dequantizer()
    manifest_path = args.output_dir / "manifest.json"
    raw_dir = args.output_dir / "raw_quantized"
    decoded_dir = args.output_dir / "decoded_bf16"
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    gptq = config["gptq"]
    from gptqmodel import BACKEND, FORMAT, GPTQConfig, GPTQModel

    offload = args.output_dir / "offload"
    quant_config = GPTQConfig(
        bits=gptq["bits"],
        group_size=gptq["group_size"],
        sym=gptq["sym"],
        desc_act=gptq["desc_act"],
        act_group_aware=gptq["act_group_aware"],
        static_groups=gptq["static_groups"],
        true_sequential=gptq["true_sequential"],
        lm_head=gptq["lm_head"],
        mse=gptq["mse"],
        damp_percent=gptq["damp_percent"],
        damp_auto_increment=gptq["damp_auto_increment"],
        fallback=None,
        format=FORMAT.GPTQ,
        pack_dtype=torch.int32,
        pack_impl=gptq["pack_impl"],
        device="cuda:0",
        offload_to_disk=True,
        offload_to_disk_path=str(offload),
        auto_forward_data_parallel=False,
        calibration_data_device="cuda:0",
    )
    model = GPTQModel.load(
        str(args.snapshot_root),
        quantize_config=quant_config,
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    calibration = [
        {"input_ids": row, "attention_mask": torch.ones_like(row)}
        for row in tokens
    ]
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    quant_log = model.quantize(
        calibration=calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
    )
    model.save(
        str(raw_dir),
        max_shard_size="4GB",
        split_by="layer",
        safetensors_metadata={
            "experiment_id": config["experiment_id"],
            "model_revision": config["model"]["revision"],
            "calibration_token_sha256": config["calibration"]["token_tensor_sha256"],
        },
    )
    del model, calibration, tokens
    gc.collect()
    torch.cuda.empty_cache()
    raw_audit = audit_raw_artifact(raw_dir)
    if raw_audit["quantized_module_count"] != EXPECTED_LINEARS:
        raise RuntimeError("GPTQ raw artifact does not cover exactly 252 Linears")
    decoded = decode_artifact(config, raw_dir, decoded_dir)
    decoded_linears = sum(len(layer["tensors"]) for layer in decoded)
    decoded_weights = sum(item["numel"] for layer in decoded for item in layer["tensors"])
    if len(decoded) != EXPECTED_LAYERS or decoded_linears != EXPECTED_LINEARS or decoded_weights != EXPECTED_WEIGHTS:
        raise RuntimeError("GPTQ decoded BF16 coverage drifted")
    analytical = analytical_gptq_storage(config["model"])
    manifest = {
        "schema_version": 1,
        "status": "completed_pending_review",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "model": config["model"],
        "calibration": config["calibration"],
        "gptq": config["gptq"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gptqmodel": importlib.metadata.version("gptqmodel"),
            "device": torch.cuda.get_device_name(0),
            "gptq_torch_triton_dequant": os.environ["GPTQ_TORCH_TRITON_DEQUANT"],
        },
        "coverage": {
            "layer_count": len(decoded),
            "linear_count": decoded_linears,
            "quantized_weight_count": decoded_weights,
            "no_fp_fallback": True,
            "decoded_dtype": "torch.bfloat16",
            "decoded_all_finite": True,
        },
        "analytical_storage": analytical,
        "raw_artifact": raw_audit,
        "decoded_layers": decoded,
        "quantization_log": json_safe(quant_log),
    }
    atomic_json(manifest_path, manifest)
    print(f"FLUXBIN_GPTQ_W3_RESULT={manifest_path}")


if __name__ == "__main__":
    main()
