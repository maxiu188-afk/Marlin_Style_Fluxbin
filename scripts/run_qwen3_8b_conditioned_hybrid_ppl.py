#!/usr/bin/env python3
"""Frozen 8B matched PPL using the accepted 32B scorer and payload decoder."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import time
from pathlib import Path

import torch
from fluxbin_style import atomic_json, sha256_file
from fluxbin_style.qwen3_8b import validate_architecture
from run_qwen3_full_hessian_obq_s8_ppl import (
    apply_arm, validate_arm_ledger, validate_matched_targets,
)
from run_qwen3_two_base_rank1_s8_ppl import load_protocol, score_model, validate_runtime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/evaluation/qwen3_8b_wikitext2_conditioned_hybrid_v1.json'
SOURCE_FILES = (
    'scripts/run_qwen3_8b_conditioned_hybrid_ppl.py',
    'scripts/run_qwen3_full_hessian_obq_s8_ppl.py',
    'scripts/run_qwen3_two_base_rank1_s8_ppl.py',
    *tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT / 'src/fluxbin_style').glob('*.py'))),
)


def validate_config(config):
    if config != json.loads(CONFIG.read_text()):
        raise ValueError('frozen 8B PPL config drifted')


def assess_quality(arms, expected_transitions, gate):
    valid = set(arms) == {'bf16', 'hybrid_s8'} and all(
        a['metrics_valid'] and a['scored_transition_count'] == expected_transitions
        and a['perplexity'] is not None and math.isfinite(a['perplexity'])
        and a['perplexity'] > 0 and a['mean_nll'] is not None
        and math.isfinite(a['mean_nll']) for a in arms.values()
    )
    comparison = {}
    for name in ('hybrid_s8',):
        gap = arms[name]['perplexity'] / arms['bf16']['perplexity'] - 1 if valid else None
        comparison[name] = {
            'relative_ppl_gap_vs_matched_bf16': gap,
            'quality_gate_passed': valid and arms[name]['perplexity'] <= arms['bf16']['perplexity'] * (1 + gate['max_relative_gap']),
            'deployment_explicitly_blocked': valid and arms[name]['perplexity'] > arms['bf16']['perplexity'] * (1 + gate['deployment_block_relative_gap']),
        }
    return valid, comparison


def preflight(config, args):
    validate_config(config)
    if sha256_file(args.full_acceptance) != config['accepted_full_review_sha256']:
        raise ValueError('full artifact acceptance hash drifted')
    review = json.loads(args.full_acceptance.read_text())
    if review.get('status') != 'passed' or review['arms']['hybrid_s8']['result_sha256'] != config['decomposition']['arms']['hybrid_s8']['accepted_result_sha256']:
        raise ValueError('full artifacts are not accepted')
    if args.snapshot_root.name != config['model']['revision']:
        raise ValueError('snapshot revision drifted')
    validate_architecture(json.loads((args.snapshot_root / 'config.json').read_text()))
    for name, digest in config['model_preflight_files'].items():
        if sha256_file(args.snapshot_root / name) != digest:
            raise ValueError(f'snapshot content drifted: {name}')
    blocks, _ = load_protocol(config, args)
    results, records = {}, {}
    for arm, prefix in (('hybrid_s8', 'hybrid'),):
        results[arm], records[arm] = validate_arm_ledger(
            config, arm=arm, result_path=getattr(args, prefix + '_result'),
            source_manifest_path=getattr(args, prefix + '_source_manifest'),
            artifact_dir=getattr(args, prefix + '_artifact_dir'),
        )
    return blocks, results, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG)
    for name in ('snapshot-root', 'protocol-manifest', 'token-artifact', 'full-acceptance',
                 'hybrid-result', 'hybrid-source-manifest', 'hybrid-artifact-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    device = validate_runtime(config)
    blocks, results, records = preflight(config, args)
    source = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    if args.validate_only:
        print('FLUXBIN_8B_PPL_PREFLIGHT=passed', flush=True)
        return
    print('FLUXBIN_8B_PPL_PREFLIGHT=passed', flush=True)
    from transformers import AutoModelForCausalLM
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot_root, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=config['evaluation']['attention_implementation'],
    ).to(device).eval()
    validate_architecture(model.config.to_dict())
    model.config.use_cache = False
    arms, materialization = {}, {}
    for arm in config['evaluation']['arms']:
        if arm != 'bf16':
            materialization[arm] = apply_arm(
                model, records[arm], arm=arm, device=device,
                group_size=config['decomposition']['group_size'],
                columns_per_group=config['decomposition']['columns_per_group'],
            )
        arms[arm] = score_model(model, blocks, arm=arm, device=device,
                               logit_chunk_tokens=config['evaluation']['logit_chunk_tokens'])
        atomic_json(args.output.with_suffix('.progress.json'), {
            'status': 'in_progress', 'config_sha256': sha256_file(args.config),
            'source_files_sha256': source, 'arms': arms, 'materialization': materialization,
        })
        print(f'FLUXBIN_{arm.upper()}_PPL={arms[arm]["perplexity"]}', flush=True)
    valid, comparison = assess_quality(arms, config['accepted_protocol']['scored_transition_count'], config['quality_gate'])
    coverage = all(
        m['layer_count'] == 36 and m['tensor_count'] == 252
        and m['parameter_count'] == 6945767424
        and m['all_payload_and_internal_tensor_hashes_validated']
        for m in materialization.values()
    ) and len(materialization) == 1
    if source != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError('evaluation source changed during run')
    atomic_json(args.output, {
        'schema_version': 2, 'status': 'completed_pending_review' if valid and coverage else 'completed_invalid_metrics',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope': config['scope'], 'model': config['model'],
        'config_sha256': sha256_file(args.config), 'source_files_sha256': source,
        'accepted_full_review_sha256': config['accepted_full_review_sha256'],
        'protocol': config['accepted_protocol'], 'evaluation': config['evaluation'],
        'baseline_policy': config['baseline_policy'], 'arms': arms, 'materialization': materialization,
        'comparison': comparison, 'quality_gate': config['quality_gate'],
        'acceptance_checks': {'metrics_and_counts_valid': valid, 'complete_arm_coverage': coverage,
                              'snapshot_and_payload_hashes_validated': True, 'same_run_bf16': True},
        'runtime': {**config['execution'], 'python': platform.python_version()},
        'elapsed_seconds': time.monotonic() - started,
        'backend_auto_launch': False, 'distillation_auto_launch': False, 'next_stage': 'not_launched',
    })
    print(f'FLUXBIN_PPL_RESULT={args.output}', flush=True)
    if not valid or not coverage:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
