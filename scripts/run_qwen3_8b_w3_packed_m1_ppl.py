#!/usr/bin/env python3
"""Measure full frozen WikiText-2 PPL through the final fused M=1 W3 route."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from fluxbin_style.evaluation import atomic_json, sha256_file
from fluxbin_style.full_model_trial import packed_modules
from fluxbin_style.fused_modules import fused_qwen3_modules
from fluxbin_style.static_decode import prepared_linears
from fluxbin_style.w3_lut_deployment import w3_lut_extension_build

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_8b_w3_full_m1_trial import load_model  # noqa: E402
from run_qwen3_two_base_rank1_s8_ppl import load_protocol  # noqa: E402
sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_w3_packed_m1_ppl_v1.json"
ARMS = ("original_bf16", "decoded_w3_bf16", "packed_w3_inline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--w3-layout-root", type=Path, required=True)
    parser.add_argument("--w3-layout-manifest-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--prewarm-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    evaluation = config.get("evaluation", {})
    acceptance = config.get("acceptance", {})
    expected = {
        "arms": list(ARMS),
        "packed_route": "structural",
        "fused_nonlinear_modules": {"rms_norm": True, "rope": True},
        "batch_size": 1,
        "tokens_per_model_call": 1,
        "teacher_forcing": True,
        "use_cache": True,
        "cache_reset_each_block": True,
        "logits_to_keep": 1,
        "nll_logits_dtype": "torch.float32",
        "attention_implementation": "sdpa",
        "tf32_allowed": False,
    }
    if config.get("schema_version") != 1 or config.get("experiment_id") != "qwen3-8b-w3-packed-m1-ppl-v1":
        raise ValueError("packed M=1 PPL protocol identity drifted")
    for name, value in expected.items():
        if evaluation.get(name) != value:
            raise ValueError(f"packed M=1 PPL evaluator drifted: {name}")
    if (
        acceptance.get("required_valid_arms") != len(ARMS)
        or acceptance.get("required_scored_transition_count_per_arm")
        != config["accepted_protocol"]["scored_transition_count"]
        or acceptance.get("required_packed_linear_count") != 252
        or acceptance.get("quality_effect_threshold") is not None
        or acceptance.get("decision") != "manual_effect_size_review"
        or acceptance.get("backend_auto_launch") is not False
        or acceptance.get("kernel_auto_change") is not False
    ):
        raise ValueError("PPL acceptance policy drifted")


def validate_runtime(config: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python runtime drifted: {platform.python_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("packed M=1 PPL requires NVIDIA CUDA")
    torch.cuda.set_device(0)
    expected = config["execution"]
    device_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if device_name not in expected["quality_device_names"] or capability != expected["compute_capability"]:
        raise RuntimeError(f"formal A100 device gate failed: {device_name}, {capability}")
    versions = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "safetensors": __import__("safetensors").__version__,
    }
    for name, value in versions.items():
        if value != expected[name]:
            raise RuntimeError(f"{name} runtime drifted: {value}")
    return torch.device("cuda:0"), {
        **versions,
        "python": platform.python_version(),
        "torch_cuda_runtime": torch.version.cuda,
        "device": device_name,
        "compute_capability": capability,
    }


def validate_snapshot(config: dict[str, Any], snapshot_root: Path) -> None:
    model = config["model"]
    if snapshot_root.name != model["revision"]:
        raise ValueError("snapshot revision drifted")
    for name, digest in model["model_preflight_files"].items():
        if sha256_file(snapshot_root / name) != digest:
            raise ValueError(f"snapshot file drifted: {name}")


def validate_environment(path: Path) -> dict[str, Any]:
    environment = json.loads(path.read_text(encoding="utf-8"))
    sources = {
        str(source.relative_to(ROOT)): sha256_file(source)
        for source in sorted((ROOT / "src/fluxbin_style").rglob("*"))
        if source.suffix in {".py", ".cu", ".cuh"}
    }
    if environment.get("status") != "ready_for_gpu_trial":
        raise ValueError("environment is not ready_for_gpu_trial")
    if environment.get("source_sha256") != sources:
        raise ValueError("capture a fresh environment after source changes")
    return {"record": environment, "source_sha256": sources}


def validate_prewarm_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prewarmed":
        raise ValueError("W3 prewarm manifest is not complete")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        raise ValueError("PPL prewarm must contain exactly three production variants")
    observed = {(item.get("row_tile"), item.get("uniform_index_probe")): item for item in variants}
    if set(observed) != {(256, False), (512, False), (1024, False)}:
        raise ValueError("PPL prewarm variant inventory drifted")
    for row_tile in (256, 512, 1024):
        record = observed[row_tile, False]
        kwargs, contract = w3_lut_extension_build(row_tile, False)
        if record.get("contract") != contract:
            raise ValueError(f"prewarm build contract drifted for R{row_tile}")
        if record.get("build_directory") != kwargs.get("build_directory"):
            raise ValueError(f"prewarm build directory drifted for R{row_tile}")
        objects = record.get("shared_objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"prewarm shared-object inventory missing for R{row_tile}")
        for item in objects:
            object_path = Path(item["path"])
            if (
                not object_path.is_file()
                or object_path.stat().st_size != item["bytes"]
                or sha256_file(object_path) != item["sha256"]
            ):
                raise ValueError(f"prewarm shared object drifted: {object_path}")


def cache_length(cache) -> int:
    value = cache.get_seq_length()
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def finite_perplexity(mean_nll: float | None) -> float | None:
    if mean_nll is None or not math.isfinite(mean_nll):
        return None
    try:
        value = math.exp(mean_nll)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@torch.inference_mode()
def packed_route_smoke(model: torch.nn.Module, token: int, device: torch.device) -> dict[str, Any] | None:
    modules = packed_modules(model)
    if not modules:
        return None
    for module in modules.values():
        module.route_counts = {"packed_m1": 0, "dense_fallback": 0}
        module._decode_audit = True
    try:
        model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            use_cache=True,
            logits_to_keep=1,
        )
    finally:
        for module in modules.values():
            module._decode_audit = False
    routes = {name: dict(module.route_counts) for name, module in modules.items()}
    passed = len(routes) == 252 and all(
        route == {"packed_m1": 1, "dense_fallback": 0} for route in routes.values()
    )
    if not passed:
        raise RuntimeError("packed M=1 route smoke failed")
    return {"passed": True, "linear_count": len(routes), "calls_per_linear": 1}


@torch.inference_mode()
def score_m1_ppl(
    model: torch.nn.Module,
    blocks: list[list[int]],
    *,
    arm: str,
    device: torch.device,
    expected_packed_linears: int,
) -> dict[str, Any]:
    """Score every transition with a one-token cached forward call."""
    if model.training:
        raise ValueError("PPL model must be in eval mode")
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    block_nll: list[float] = []
    block_hashes: list[str] = []
    nonfinite_blocks: list[int] = []
    scored_transitions = 0
    logits_dtype = None
    with prepared_linears(model, torch.bfloat16) as modules:
        if len(modules) != expected_packed_linears:
            raise RuntimeError(
                f"packed Linear count drifted for {arm}: {len(modules)} != {expected_packed_linears}"
            )
        route_smoke = packed_route_smoke(model, blocks[0][0], device)
        for block_index, block in enumerate(blocks):
            if len(block) < 2:
                raise ValueError("PPL block must contain at least two tokens")
            tokens = torch.tensor(block, dtype=torch.long, device=device)
            nll_values = torch.empty(len(block) - 1, dtype=torch.float32, device=device)
            cache = None
            for position in range(len(block) - 1):
                output = model(
                    input_ids=tokens[position : position + 1].view(1, 1),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                logits = output.logits[0, -1].float()
                logits_dtype = str(output.logits.dtype)
                label = tokens[position + 1]
                nll_values[position] = torch.logsumexp(logits, dim=0) - logits[label]
            if cache_length(cache) != len(block) - 1:
                raise RuntimeError("KV cache length drifted during M=1 PPL")
            total = float(nll_values.double().sum())
            if not math.isfinite(total):
                nonfinite_blocks.append(block_index)
                break
            block_nll.append(total)
            block_hashes.append(sha256_file_tensor(nll_values))
            scored_transitions += len(block) - 1
            del tokens, nll_values, cache, output, logits
            if (block_index + 1) % 5 == 0 or block_index + 1 == len(blocks):
                print(
                    f"FLUXBIN_{arm.upper()}_M1_PPL_PROGRESS={block_index + 1}/{len(blocks)}",
                    flush=True,
                )
    if cuda:
        torch.cuda.synchronize(device)
    total_nll = math.fsum(block_nll) if not nonfinite_blocks else None
    mean_nll = total_nll / scored_transitions if total_nll is not None else None
    perplexity = finite_perplexity(mean_nll)
    return {
        "arm": arm,
        "status": "completed" if perplexity is not None and not nonfinite_blocks else "invalid",
        "execution": "teacher_forced_cached_decode_m1",
        "tokens_per_model_call": 1,
        "block_count": len(block_nll),
        "scored_transition_count": scored_transitions,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "metrics_valid": perplexity is not None and not nonfinite_blocks,
        "nonfinite_block_indices": nonfinite_blocks,
        "block_nll_sha256": block_hashes,
        "logits_dtype": logits_dtype,
        "nll_logits_dtype": "torch.float32",
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "packed_route_smoke": route_smoke,
        "packed_linear_count": len(packed_modules(model)),
        "dense_fallback_permitted_during_scoring": False,
    }


def sha256_file_tensor(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_ppl = candidate["perplexity"]
    baseline_ppl = baseline["perplexity"]
    return {
        "candidate": candidate["arm"],
        "baseline": baseline["arm"],
        "ppl_delta": candidate_ppl - baseline_ppl,
        "ppl_relative_percent": (candidate_ppl / baseline_ppl - 1.0) * 100.0,
        "mean_nll_delta": candidate["mean_nll"] - baseline["mean_nll"],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    validate_snapshot(config, args.snapshot_root)
    blocks, protocol_manifest = load_protocol(config, args)
    if sha256_file(args.w3_layout_root / "manifest.json") != args.w3_layout_manifest_sha256:
        raise ValueError("W3 layout manifest drifted")
    environment = validate_environment(args.environment)
    validate_prewarm_manifest(args.prewarm_manifest)
    device, runtime = validate_runtime(config)
    provenance = {
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "environment_sha256": sha256_file(args.environment),
        "prewarm_manifest_sha256": sha256_file(args.prewarm_manifest),
        "w3_layout_manifest_sha256": args.w3_layout_manifest_sha256,
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
        "token_artifact_sha256": sha256_file(args.token_artifact),
        "source_sha256": environment["source_sha256"],
    }
    run_path = args.output_dir / "run.json"
    if args.resume:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("provenance") != provenance:
            raise ValueError("resume provenance drifted")
    else:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        (args.output_dir / "arms").mkdir(parents=True)
        run = {
            "schema_version": 1,
            "status": "running",
            "experiment_id": config["experiment_id"],
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "provenance": provenance,
            "runtime": runtime,
            "protocol": config["accepted_protocol"],
            "protocol_manifest_status": protocol_manifest["status"],
            "completed_arms": [],
        }
        atomic_json(run_path, run)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    route_by_arm = {"packed_w3_inline": config["evaluation"]["packed_route"]}
    metrics: dict[str, Any] = {}
    fusion = ExitStack()
    try:
        run["fused_nonlinear_modules"] = fusion.enter_context(
            fused_qwen3_modules(**config["evaluation"]["fused_nonlinear_modules"])
        )
        for arm in ARMS:
            arm_path = args.output_dir / "arms" / f"{arm}.json"
            if args.resume and arm_path.is_file():
                record = json.loads(arm_path.read_text(encoding="utf-8"))
                if record.get("provenance") != provenance or record.get("metrics", {}).get("metrics_valid") is not True:
                    raise ValueError(f"invalid resumable arm: {arm}")
                metrics[arm] = record["metrics"]
                continue
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            model, coverage = load_model(
                args.snapshot_root,
                args.w3_layout_root,
                args.w3_layout_manifest_sha256,
                arm,
                route_by_arm,
            )
            model.config.use_cache = True
            expected_packed = 252 if arm == "packed_w3_inline" else 0
            metrics[arm] = score_m1_ppl(
                model,
                blocks,
                arm=arm,
                device=device,
                expected_packed_linears=expected_packed,
            )
            atomic_json(
                arm_path,
                {"schema_version": 1, "provenance": provenance, "coverage": coverage, "metrics": metrics[arm]},
            )
            run["completed_arms"] = [name for name in ARMS if (args.output_dir / "arms" / f"{name}.json").is_file()]
            atomic_json(run_path, run)
            print(f"FLUXBIN_{arm.upper()}_M1_PPL={metrics[arm]['perplexity']}", flush=True)
            del model
            torch.cuda.empty_cache()

        expected_transitions = config["acceptance"]["required_scored_transition_count_per_arm"]
        validity = {
            arm: bool(
                metrics[arm]["metrics_valid"]
                and metrics[arm]["block_count"] == config["accepted_protocol"]["full_block_count"]
                and metrics[arm]["scored_transition_count"] == expected_transitions
                and (
                    arm != "packed_w3_inline"
                    or metrics[arm]["packed_route_smoke"] == {
                        "passed": True,
                        "linear_count": 252,
                        "calls_per_linear": 1,
                    }
                )
            )
            for arm in ARMS
        }
        summary = {
            "schema_version": 1,
            "status": "completed_pending_effect_size_review" if all(validity.values()) else "completed_invalid_metrics",
            "experiment_id": config["experiment_id"],
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "provenance": provenance,
            "runtime": runtime,
            "scope": config["scope"],
            "arms": metrics,
            "validity_gates": validity,
            "comparisons": {
                "decoded_w3_vs_original_bf16": comparison(metrics["decoded_w3_bf16"], metrics["original_bf16"]),
                "packed_w3_vs_decoded_w3": comparison(metrics["packed_w3_inline"], metrics["decoded_w3_bf16"]),
                "packed_w3_vs_original_bf16": comparison(metrics["packed_w3_inline"], metrics["original_bf16"]),
            },
            "quality_effect_threshold": None,
            "decision": "manual_effect_size_review",
            "performance_claim": None,
            "next_stage": "not_launched",
        }
        atomic_json(args.output_dir / "summary.json", summary)
        run.update(status=summary["status"], finished_at_utc=summary["created_at_utc"])
        atomic_json(run_path, run)
        print(summary["status"])
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(run_path, run)
        raise
    finally:
        fusion.close()


if __name__ == "__main__":
    main()
