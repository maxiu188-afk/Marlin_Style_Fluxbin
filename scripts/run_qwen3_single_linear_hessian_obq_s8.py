#!/usr/bin/env python3
"""One calibrated Qwen3 Linear gate for pure and hybrid FluxBin-style OBQ."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fluxbin_style import (
    InputHessianAccumulator,
    TWO_BASE_PACKED_FORMAT,
    TwoBaseRankOneOptimizationConfig,
    atomic_json,
    invert_hessian,
    pack_two_bases,
    quantize_hybrid_two_base_obq,
    quantize_pure_two_base_obq,
    sha256_file,
    tensor_sha256,
)


class _CaptureComplete(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def validate_runtime(config: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    expected = config["execution"]
    if torch.cuda.get_device_name(0) != expected["device_name"]:
        raise RuntimeError("unexpected GPU")
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
    return torch.device("cuda:0")


def solver_config(value: dict[str, Any]) -> TwoBaseRankOneOptimizationConfig:
    return TwoBaseRankOneOptimizationConfig(**value)


def load_calibration(
    config: dict[str, Any],
    manifest_path: Path,
    tokens_path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["calibration"]
    if manifest.get("status") != "passed":
        raise ValueError("calibration artifact is not passed")
    if manifest.get("artifact_id") != expected["artifact_id"]:
        raise ValueError("calibration artifact id drifted")
    if manifest["config"]["dataset"]["repo_id"] != expected["dataset"]:
        raise ValueError("calibration dataset drifted")
    sampling = manifest["config"]["sampling"]
    if sampling["sequences"] != expected["sequences"]:
        raise ValueError("calibration sequence count drifted")
    if sampling["sequence_length"] != expected["sequence_length"]:
        raise ValueError("calibration sequence length drifted")
    if sha256_file(tokens_path) != manifest["tokens"]["file_sha256"]:
        raise ValueError("calibration token file hash drifted")
    with safe_open(tokens_path, framework="pt", device="cpu") as artifact:
        if set(artifact.keys()) != {"token_ids"}:
            raise ValueError("calibration token inventory drifted")
        tokens = artifact.get_tensor("token_ids")
    if list(tokens.shape) != [expected["sequences"], expected["sequence_length"]]:
        raise ValueError("calibration token shape drifted")
    if tensor_sha256(tokens) != manifest["tokens"]["tensor_sha256"]:
        raise ValueError("calibration token tensor hash drifted")
    return tokens.to(torch.int64), manifest


def capture_target_hessian(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    target_module: torch.nn.Linear,
    device: torch.device,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    layers = model.model.layers
    sample_count, sequence_length = tokens.shape
    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    inputs = torch.empty(
        (sample_count, sequence_length, hidden_size),
        dtype=dtype,
        device=device,
    )
    captured_kwargs: dict[str, Any] = {}
    captured = 0

    class CaptureFirstLayer(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
            nonlocal captured
            if captured >= sample_count:
                raise RuntimeError("received too many calibration sequences")
            inputs[captured].copy_(hidden_states[0])
            if not captured_kwargs:
                captured_kwargs.update(kwargs)
            captured += 1
            raise _CaptureComplete()

    original_first = layers[0]
    previous_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    layers[0] = CaptureFirstLayer(original_first)
    try:
        with torch.inference_mode():
            for sample_index in range(sample_count):
                try:
                    model(
                        input_ids=tokens[sample_index : sample_index + 1].to(device),
                        use_cache=False,
                    )
                except _CaptureComplete:
                    pass
                if (sample_index + 1) % 32 == 0:
                    print(
                        f"FLUXBIN_OBQ_INPUT_CAPTURE={sample_index + 1}/{sample_count}",
                        flush=True,
                    )
    finally:
        layers[0] = original_first
        model.config.use_cache = previous_cache
    if captured != sample_count:
        raise RuntimeError("failed to capture every calibration sequence")

    accumulator = InputHessianAccumulator(target_module.in_features, device=device)

    def collect(
        _module: torch.nn.Module,
        module_inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        accumulator.add(module_inputs[0])

    handle = target_module.register_forward_hook(collect)
    try:
        with torch.inference_mode():
            for sample_index in range(sample_count):
                original_first(inputs[sample_index : sample_index + 1], **captured_kwargs)
                if (sample_index + 1) % 16 == 0:
                    print(
                        f"FLUXBIN_OBQ_HESSIAN_CAPTURE={sample_index + 1}/{sample_count}",
                        flush=True,
                    )
    finally:
        handle.remove()
    hessian = accumulator.value()
    if accumulator.sample_count != sample_count * sequence_length:
        raise RuntimeError("Hessian activation-row count drifted")
    capture = {
        "calibration_sequences": sample_count,
        "sequence_length": sequence_length,
        "activation_rows": accumulator.sample_count,
        "hidden_state_dtype": str(dtype),
        "hessian_dtype": str(hessian.dtype),
        "hessian_shape": list(hessian.shape),
        "hessian_sha256": tensor_sha256(hessian),
    }
    return hessian, accumulator.sample_count, capture


def reconstruction_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    hessian: torch.Tensor,
) -> dict[str, float]:
    error = target.to(torch.float32) - reconstruction.to(torch.float32)
    squared_error = error.square().sum(dtype=torch.float64)
    target_squared_norm = target.square().sum(dtype=torch.float64)
    quadratic = 0.5 * (error @ hessian * error).sum(dtype=torch.float64)
    metrics = {
        "squared_error": float(squared_error.item()),
        "mse": float((squared_error / error.numel()).item()),
        "relative_frobenius_error": float(
            (squared_error.sqrt() / target_squared_norm.sqrt()).item()
        ),
        "calibration_mean_output_squared_error": float(
            (quadratic / target.shape[0]).item()
        ),
        "calibration_total_output_squared_error": float(quadratic.item()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("reconstruction metrics contain non-finite values")
    return metrics


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, temporary)
    os.replace(temporary, path)


def convergence_summary(groups) -> dict[str, Any]:
    global_reasons = Counter(group.global_converged_reason for group in groups)
    refinement_reasons = Counter(
        group.refinement_converged_reason
        for group in groups
        if group.refinement_converged_reason is not None
    )
    return {
        "group_count": len(groups),
        "global_converged_reasons": dict(sorted(global_reasons.items())),
        "global_max_iterations": max(group.global_iterations for group in groups),
        "refinement_converged_reasons": dict(sorted(refinement_reasons.items())),
        "refinement_max_iterations": max(
            (
                group.refinement_iterations
                for group in groups
                if group.refinement_iterations is not None
            ),
            default=None,
        ),
        "propagated_update_squared_norm": math.fsum(
            group.propagated_update_squared_norm for group in groups
        ),
        "groups": [asdict(group) for group in groups],
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM

    for path in (args.payload, args.output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("unsupported schema_version")
    if args.snapshot_root.name != config["model"]["revision"]:
        raise ValueError("model snapshot revision drifted")
    device = validate_runtime(config)
    torch.manual_seed(config["seed"])
    tokens, calibration_manifest = load_calibration(
        config,
        args.calibration_manifest,
        args.calibration_tokens,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()
    target_name = config["model"]["target_tensor"]
    module_name = target_name.removesuffix(".weight")
    target_module = model.get_submodule(module_name)
    if not isinstance(target_module, torch.nn.Linear):
        raise TypeError("target module is not Linear")
    target = target_module.weight.detach().to(torch.float32).clone()
    if list(target.shape) != config["model"]["target_shape"]:
        raise ValueError("target shape drifted")
    if tensor_sha256(target_module.weight.detach().cpu()) != config["model"][
        "target_tensor_sha256"
    ]:
        raise ValueError("target tensor hash drifted")
    hessian, activation_rows, capture = capture_target_hessian(
        model,
        tokens,
        target_module=target_module,
        device=device,
    )
    del model, target_module, tokens
    torch.cuda.empty_cache()

    inverse_result = invert_hessian(
        hessian,
        damp_percent=config["calibration"]["damp_percent"],
    )
    global_config = solver_config(config["global_solver"])
    refinement_config = solver_config(config["refinement_solver"])
    pure = quantize_pure_two_base_obq(
        target,
        inverse_result.inverse,
        group_size=config["algorithm"]["global_group_size"],
        config=global_config,
    )
    hybrid = quantize_hybrid_two_base_obq(
        target,
        inverse_result.inverse,
        group_size=config["algorithm"]["global_group_size"],
        columns_per_group=config["algorithm"]["hybrid_arm"][
            "residual_columns_per_group"
        ],
        global_config=global_config,
        refinement_config=refinement_config,
    )
    pure_weight = pure.decomposition.reconstruct()
    hybrid_weight = hybrid.decomposition.reconstruct()
    pure_metrics = reconstruction_metrics(target, pure_weight, hessian)
    hybrid_metrics = reconstruction_metrics(target, hybrid_weight, hessian)

    hybrid_delta = (
        hybrid.decomposition.reconstruct_grouped()
        - hybrid.decomposition.global_decomposition.reconstruct_grouped()
    )
    selected_mask = torch.zeros(
        (
            hybrid.decomposition.global_decomposition.num_groups,
            hybrid.decomposition.global_decomposition.group_size,
        ),
        dtype=torch.bool,
        device=device,
    )
    selected_mask.scatter_(1, hybrid.decomposition.selected_indices, True)
    outside = ~selected_mask.unsqueeze(0).expand(target.shape[0], -1, -1)
    maximum_outside_delta = float(hybrid_delta[outside].abs().max().item())
    if maximum_outside_delta != 0:
        raise RuntimeError("hybrid refinement changed an unselected column")

    refinement = hybrid.decomposition.refinement_decomposition
    payload_tensors = {
        "pure_global_sign_codes": pack_two_bases(pure.decomposition.bases),
        "pure_global_row_scales": pure.decomposition.row_scales,
        "pure_global_column_scales": pure.decomposition.column_scales,
        "hybrid_global_sign_codes": pack_two_bases(
            hybrid.decomposition.global_decomposition.bases
        ),
        "hybrid_global_row_scales": hybrid.decomposition.global_decomposition.row_scales,
        "hybrid_global_column_scales": hybrid.decomposition.global_decomposition.column_scales,
        "hybrid_refinement_indices": hybrid.decomposition.selected_indices.to(torch.int16),
        "hybrid_refinement_sign_codes": pack_two_bases(refinement.bases),
        "hybrid_refinement_row_scales": refinement.row_scales,
        "hybrid_refinement_column_scales": refinement.column_scales,
    }
    atomic_safetensors(args.payload, payload_tensors)
    payload_hashes = {name: tensor_sha256(value) for name, value in payload_tensors.items()}
    with safe_open(args.payload, framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(payload_tensors):
            raise RuntimeError("payload tensor inventory drifted")
        for name, expected_hash in payload_hashes.items():
            if tensor_sha256(reopened.get_tensor(name)) != expected_hash:
                raise RuntimeError(f"payload tensor hash drifted: {name}")

    result = {
        "schema_version": 2,
        "status": "completed_pending_review",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": sha256_file(args.config),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "model": {
            **config["model"],
            "observed_target_tensor_sha256": tensor_sha256(target.cpu()),
        },
        "calibration": {
            **capture,
            "artifact_manifest_sha256": sha256_file(args.calibration_manifest),
            "token_artifact_sha256": sha256_file(args.calibration_tokens),
            "dataset_revision": calibration_manifest["config"]["dataset"]["revision"],
            "hessian_definition": config["calibration"]["hessian_definition"],
            "inverse_method": config["calibration"]["inverse_method"],
            "damp_percent": inverse_result.damp_percent,
            "damping": inverse_result.damping,
            "activation_rows": activation_rows,
        },
        "contract": {
            **config["algorithm"],
            "payload_format": TWO_BASE_PACKED_FORMAT,
            "stored_arms": ["pure_two_base_obq", "hessian_salient_hybrid_s8_obq"],
        },
        "reconstruction": {
            "pure_two_base_obq": pure_metrics,
            "hessian_salient_hybrid_s8_obq": hybrid_metrics,
            "hybrid_relative_calibration_loss_reduction_vs_pure": (
                pure_metrics["calibration_total_output_squared_error"]
                - hybrid_metrics["calibration_total_output_squared_error"]
            )
            / pure_metrics["calibration_total_output_squared_error"],
            "hybrid_maximum_delta_outside_selected_columns": maximum_outside_delta,
        },
        "solver": {
            "pure": convergence_summary(pure.groups),
            "hybrid": convergence_summary(hybrid.groups),
        },
        "payload": {
            "path": args.payload.name,
            "bytes": args.payload.stat().st_size,
            "sha256": sha256_file(args.payload),
            "tensor_sha256": payload_hashes,
            "round_trip_verified": True,
        },
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.monotonic() - started,
        },
        "decision": {
            "execution_valid": True,
            "metrics_finite": all(
                math.isfinite(value)
                for metrics in (pure_metrics, hybrid_metrics)
                for value in metrics.values()
            ),
            "pure_arm_preserved_independently": True,
            "manual_review_required": True,
            "full_model_auto_launch": False,
            "ppl_auto_launch": False,
            "backend_auto_launch": False,
        },
        "next_stage": "not_launched",
    }
    atomic_json(args.output, result)
    print(f"FLUXBIN_OBQ_RESULT={args.output}")
    print(f"FLUXBIN_OBQ_PAYLOAD={args.payload}")


if __name__ == "__main__":
    main()
