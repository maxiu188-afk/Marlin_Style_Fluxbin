#!/usr/bin/env python3
"""Paired local-block gate for offline rotation before full Qwen3-8B GPTQ.

The probe compares unrotated and offline-rotated H2.50 on fixed decoder
layers.  Each arm uses its own pre-quantization block output as the local
reference; this is not a rotated-BF16 PPL arm.  No payload or model checkpoint
is retained, and a passing result only makes a manual full-model launch
eligible.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from fluxbin_style import (
    QWEN3_LINEAR_MODULES,
    atomic_json,
    capture_first_layer_inputs,
    capture_linear_hessian,
    expected_qwen3_linear_shape,
    hidden_states,
    propagate_layer_inputs,
    sha256_file,
    tensor_sha256,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_8b_hierarchical_w2_gptq import (  # noqa: E402
    EXECUTION_POLICIES,
    FORMAL_EXECUTION_POLICY,
    GROUPS,
    quantize_module,
)
from run_qwen3_8b_hierarchical_w2_rotated_gptq import (  # noqa: E402
    ARM,
    rotate_model,
    runtime_identity,
    validate_config,
    validate_inputs,
    validate_runtime,
)
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
ARMS = ("unrotated_h2_50", "rotated_h2_50")
STAGES = ("layer0", "representative")
PROBE_SOURCE_FILES = (
    "src/fluxbin_style/__init__.py",
    "src/fluxbin_style/evaluation.py",
    "src/fluxbin_style/hessian_obq.py",
    "src/fluxbin_style/hierarchical_w2.py",
    "src/fluxbin_style/offline_rotation.py",
    "src/fluxbin_style/qwen3.py",
    "src/fluxbin_style/qwen3_sequential.py",
    "scripts/run_qwen3_8b_hierarchical_w2_gptq.py",
    "scripts/run_qwen3_8b_hierarchical_w2_rotated_gptq.py",
    "scripts/run_qwen3_8b_hierarchical_w2_rotation_probe.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-policy", choices=EXECUTION_POLICIES, default=FORMAL_EXECUTION_POLICY
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def source_hash() -> tuple[str, dict[str, str]]:
    files = {name: sha256_file(ROOT / name) for name in PROBE_SOURCE_FILES}
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode())
        digest.update(bytes.fromhex(value))
    return digest.hexdigest(), files


def validate_probe_config(config: dict[str, Any]) -> None:
    validate_config(config)
    probe = config.get("probe", {})
    if probe.get("calibration_prefix_samples") != 32:
        raise ValueError("probe calibration prefix drifted")
    if list(probe.get("stages", {})) != list(STAGES):
        raise ValueError("probe stage inventory drifted")
    expected = {
        "layer0": {
            "execution_layers": [0],
            "evaluation_layers": [0],
            "max_aggregate_block_error_ratio": 0.95,
            "min_layer_wins": 1,
            "max_layer_error_ratio": 0.95,
            "min_linear_wins": 4,
            "max_down_proj_error_ratio": 1.25,
            "pass_action": "advance_to_representative_probe",
            "fail_action": "stop_without_full_model",
        },
        "representative": {
            "execution_layers": [17, 35],
            "evaluation_layers": [0, 17, 35],
            "max_aggregate_block_error_ratio": 0.85,
            "min_layer_wins": 2,
            "max_layer_error_ratio": 1.05,
            "min_linear_wins": 14,
            "max_down_proj_error_ratio": 1.05,
            "pass_action": "eligible_for_manual_full_model_launch",
            "fail_action": "stop_without_full_model",
        },
    }
    if probe["stages"] != expected:
        raise ValueError("probe stages or gates drifted")
    if probe.get("full_model_auto_launch") is not False:
        raise ValueError("probe must not auto-launch the full model")
    layers = expected["representative"]["evaluation_layers"]
    if layers != sorted(set(layers)) or layers[-1] >= config["model"]["expected_hidden_layers"]:
        raise ValueError("probe layer inventory is invalid")


def load_model(config: dict[str, Any], snapshot_root: Path) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=config["evaluation"]["attention_implementation"],
    ).eval()
    model.config.use_cache = False
    return model


@torch.no_grad()
def block_output_metrics(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    references: torch.Tensor,
    forward_kwargs: dict[str, Any],
) -> dict[str, float]:
    squared_error = 0.0
    reference_energy = 0.0
    maximum_absolute_error = 0.0
    for sample_index in range(inputs.shape[0]):
        actual = hidden_states(
            layer(inputs[sample_index : sample_index + 1], **forward_kwargs)
        )[0].to(torch.float32)
        reference = references[sample_index].to(torch.float32)
        difference = reference - actual
        squared_error += float(difference.square().sum(dtype=torch.float64).item())
        reference_energy += float(reference.square().sum(dtype=torch.float64).item())
        maximum_absolute_error = max(
            maximum_absolute_error, float(difference.abs().max().item())
        )
    if squared_error <= 0.0 or reference_energy <= 0.0:
        raise RuntimeError("probe block metrics must be positive")
    metrics = {
        "squared_error": squared_error,
        "reference_squared_norm": reference_energy,
        "relative_squared_error": squared_error / reference_energy,
        "relative_rmse": math.sqrt(squared_error / reference_energy),
        "maximum_absolute_error": maximum_absolute_error,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("probe block metrics are non-finite")
    return metrics


@torch.no_grad()
def run_arm(
    config: dict[str, Any],
    *,
    arm: str,
    stage: str,
    tokens: torch.Tensor,
    snapshot_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = load_model(config, snapshot_root)
    rotation = None
    if arm == "rotated_h2_50":
        rotation = rotate_model(model, config, device=device)
    elif arm != "unrotated_h2_50":
        raise ValueError(f"unsupported probe arm: {arm}")

    selected_layers = config["probe"]["stages"][stage]["execution_layers"]
    model_config = json.loads((snapshot_root / "config.json").read_text())
    for layer_index in selected_layers:
        layer = model.model.layers[layer_index]
        for name in QWEN3_LINEAR_MODULES:
            module = layer.get_submodule(name)
            if (
                not isinstance(module, torch.nn.Linear)
                or list(module.weight.shape) != expected_qwen3_linear_shape(model_config, name)
            ):
                raise ValueError(f"Linear inventory drifted: {layer_index}:{name}")

    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    model.model.layers[0].to(device)
    first = capture_first_layer_inputs(model, tokens, device=device)
    inputs, forward_kwargs = first.inputs, first.forward_kwargs
    outputs = torch.empty_like(inputs)
    model.model.layers[0].to("cpu")
    model.model.embed_tokens.to("cpu")
    model.model.rotary_emb.to("cpu")
    torch.cuda.empty_cache()

    layer_records = []
    arm_started = time.monotonic()
    selected = set(selected_layers)
    for layer_index, layer in enumerate(model.model.layers[: max(selected_layers) + 1]):
        layer.to(device)
        propagate_layer_inputs(layer, inputs, outputs, forward_kwargs)
        layer.to("cpu")
        if layer_index in selected:
            candidate = copy.deepcopy(layer).to(device).eval()
            payload_tensors: dict[str, torch.Tensor] = {}
            linears = []
            hessians = {}
            layer_started = time.monotonic()
            for group_name, source_name, module_names in GROUPS:
                capture = capture_linear_hessian(
                    candidate,
                    candidate.get_submodule(source_name),
                    inputs,
                    forward_kwargs,
                )
                hessians[group_name] = {
                    "source_module": source_name,
                    "activation_rows": capture.activation_rows,
                    "shape": list(capture.hessian.shape),
                    "diagonal_sha256": tensor_sha256(capture.hessian.diagonal()),
                }
                for module_name in module_names:
                    record = quantize_module(
                        candidate.get_submodule(module_name),
                        capture.hessian,
                        module_name=module_name,
                        variant=config["variants"][ARM],
                        config=config,
                        payload_tensors=payload_tensors,
                    )
                    linears.append(record)
                del capture
                torch.cuda.empty_cache()
            block = block_output_metrics(candidate, inputs, outputs, forward_kwargs)
            layer_records.append(
                {
                    "layer_index": layer_index,
                    "block_output": block,
                    "hessians": hessians,
                    "linears": linears,
                    "elapsed_seconds": time.monotonic() - layer_started,
                }
            )
            del candidate, payload_tensors
            torch.cuda.empty_cache()
            print(f"ROTATION_PROBE_LAYER={arm}:{layer_index}", flush=True)
        inputs, outputs = outputs, inputs
        torch.cuda.empty_cache()

    result = {
        "arm": arm,
        "rotation": rotation,
        "layers": layer_records,
        "elapsed_seconds": time.monotonic() - arm_started,
    }
    del model, inputs, outputs
    torch.cuda.empty_cache()
    return result


def normalized_linear_error(record: dict[str, Any]) -> float:
    value = record["squared_error"] / record["target_squared_norm"]
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("invalid normalized Linear error")
    return value


def evaluate_gate(
    config: dict[str, Any], stage: str, arms: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    controls = config["probe"]["stages"][stage]
    original_layers = {row["layer_index"]: row for row in arms[ARMS[0]]["layers"]}
    rotated_layers = {row["layer_index"]: row for row in arms[ARMS[1]]["layers"]}
    evaluation_layers = controls["evaluation_layers"]
    if list(original_layers) != evaluation_layers or list(rotated_layers) != evaluation_layers:
        raise ValueError("probe layer coverage drifted")

    original_sse = math.fsum(
        row["block_output"]["squared_error"] for row in original_layers.values()
    )
    original_energy = math.fsum(
        row["block_output"]["reference_squared_norm"] for row in original_layers.values()
    )
    rotated_sse = math.fsum(
        row["block_output"]["squared_error"] for row in rotated_layers.values()
    )
    rotated_energy = math.fsum(
        row["block_output"]["reference_squared_norm"] for row in rotated_layers.values()
    )
    original_relative = original_sse / original_energy
    rotated_relative = rotated_sse / rotated_energy
    aggregate_ratio = rotated_relative / original_relative

    layer_ratios = {}
    linear_ratios = {}
    down_original_error = 0.0
    down_original_norm = 0.0
    down_rotated_error = 0.0
    down_rotated_norm = 0.0
    for layer_index in evaluation_layers:
        original = original_layers[layer_index]
        rotated = rotated_layers[layer_index]
        layer_ratios[str(layer_index)] = (
            rotated["block_output"]["relative_squared_error"]
            / original["block_output"]["relative_squared_error"]
        )
        original_linears = {row["module"]: row for row in original["linears"]}
        rotated_linears = {row["module"]: row for row in rotated["linears"]}
        if list(original_linears) != list(QWEN3_LINEAR_MODULES) or list(rotated_linears) != list(QWEN3_LINEAR_MODULES):
            raise ValueError("probe Linear inventory drifted")
        for module_name in QWEN3_LINEAR_MODULES:
            key = f"{layer_index}:{module_name}"
            linear_ratios[key] = normalized_linear_error(
                rotated_linears[module_name]
            ) / normalized_linear_error(original_linears[module_name])
        original_down = original_linears["mlp.down_proj"]
        rotated_down = rotated_linears["mlp.down_proj"]
        down_original_error += original_down["squared_error"]
        down_original_norm += original_down["target_squared_norm"]
        down_rotated_error += rotated_down["squared_error"]
        down_rotated_norm += rotated_down["target_squared_norm"]
    down_ratio = (down_rotated_error / down_rotated_norm) / (
        down_original_error / down_original_norm
    )
    layer_wins = sum(value < 1.0 for value in layer_ratios.values())
    linear_wins = sum(value < 1.0 for value in linear_ratios.values())
    checks = {
        "aggregate_block_error_ratio": aggregate_ratio
        <= controls["max_aggregate_block_error_ratio"],
        "layer_wins": layer_wins >= controls["min_layer_wins"],
        "maximum_layer_error_ratio": max(layer_ratios.values())
        <= controls["max_layer_error_ratio"],
        "linear_wins": linear_wins >= controls["min_linear_wins"],
        "down_proj_error_ratio": down_ratio <= controls["max_down_proj_error_ratio"],
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "action": controls["pass_action"] if passed else controls["fail_action"],
        "checks": checks,
        "thresholds": controls,
        "aggregate": {
            "unrotated_block_relative_squared_error": original_relative,
            "rotated_block_relative_squared_error": rotated_relative,
            "rotated_to_unrotated_ratio": aggregate_ratio,
            "relative_error_reduction": 1.0 - aggregate_ratio,
        },
        "layer_error_ratios": layer_ratios,
        "layer_wins": layer_wins,
        "linear_error_ratios": linear_ratios,
        "linear_wins": linear_wins,
        "down_proj_error_ratio": down_ratio,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_probe_config(config)
    if args.stage == "layer0" and args.prior_result is not None:
        raise ValueError("layer0 probe must not receive a prior result")
    if args.stage == "representative" and args.prior_result is None:
        raise ValueError("representative probe requires the passed layer0 result")
    tokens = validate_inputs(config, args)
    prefix_samples = config["probe"]["calibration_prefix_samples"]
    tokens = tokens[:prefix_samples].contiguous()
    if args.validate_only:
        print("ROTATION_PROBE_INPUTS_PASSED; no probe launched", flush=True)
        return

    device, runtime = validate_runtime(config, execution_policy=args.execution_policy)
    runtime = {**runtime, "cuda": torch.version.cuda}
    identity = runtime_identity(runtime)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()

    prior = None
    if args.prior_result is not None:
        prior = json.loads(args.prior_result.read_text())
        if (
            prior.get("stage") != "layer0"
            or prior.get("config_sha256") != sha256_file(args.config)
            or prior.get("gate", {}).get("passed") is not True
            or prior.get("gate", {}).get("action") != "advance_to_representative_probe"
            or {key: prior.get("execution", {}).get(key) for key in identity} != identity
        ):
            raise ValueError("prior layer0 result is not eligible for representative probe")

    arms = {
        arm: run_arm(
            config,
            arm=arm,
            stage=args.stage,
            tokens=tokens,
            snapshot_root=args.snapshot_root,
            device=device,
        )
        for arm in ARMS
    }
    if prior is not None:
        for arm in ARMS:
            arms[arm]["layers"] = prior["arms"][arm]["layers"] + arms[arm]["layers"]
    gate = evaluate_gate(config, args.stage, arms)
    implementation_hash, implementation_files = source_hash()
    result = {
        "schema_version": 1,
        "status": "completed_probe_passed" if gate["passed"] else "completed_probe_stopped",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "stage": args.stage,
        "config_sha256": sha256_file(args.config),
        "implementation_sha256": implementation_hash,
        "implementation_files_sha256": implementation_files,
        "variant": config["variants"][ARM],
        "calibration": {
            "source_artifact_id": config["calibration"]["artifact_id"],
            "selection": "first_contiguous_sequences_without_resampling",
            "sample_count": prefix_samples,
            "sequence_length": config["calibration"]["sequence_length"],
            "token_prefix_sha256": tensor_sha256(tokens),
        },
        "probe": config["probe"],
        "prior_result": (
            {
                "path": str(args.prior_result),
                "sha256": sha256_file(args.prior_result),
            }
            if args.prior_result is not None
            else None
        ),
        "arms": arms,
        "gate": gate,
        "execution": {
            **identity,
            "host": platform.node(),
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "full_model_launched": False,
        "ppl_launched": False,
        "rotated_bf16_ppl_arm": False,
    }
    atomic_json(args.output, result)
    print(f"ROTATION_PROBE_RESULT={args.output}", flush=True)
    print(f"ROTATION_PROBE_GATE={gate['action']}", flush=True)


if __name__ == "__main__":
    main()
