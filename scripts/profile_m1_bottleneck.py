#!/usr/bin/env python3
"""Bounded real-weight M1 diagnostic; profile capture is never performance evidence."""
import argparse
import json
from pathlib import Path

import torch
from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing
from fluxbin_style.deployment import FIELDS, convert_artifact, decode_layout, m1_out, workspace_shape
from fluxbin_style.deployment_artifacts import load_accepted_layer
from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('self_attn.k_proj', 'self_attn.q_proj', 'mlp.gate_proj', 'mlp.down_proj')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--module', choices=MODULES, required=True)
    p.add_argument('--mode', choices=('timing', 'profile'), required=True)
    p.add_argument('--arm', choices=('v1', 'v2', 'v3', 'dense'), default='v3',
                   help='Profile mode selects one arm; timing measures all arms')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA required; no CPU/MPS substitute')
    env = json.loads(args.environment.read_text())
    sources = {str(f.relative_to(ROOT)): sha256_file(f)
               for f in sorted((ROOT/'src/fluxbin_style').rglob('*')) if f.suffix in ('.py', '.cu')}
    if env['status'] != 'ready_for_gpu_trial' or env['source_sha256'] != sources:
        raise ValueError('ready source-matched environment required')
    if (env['torch']['version'] != torch.__version__ or
        env['torch']['cuda_runtime'] != torch.version.cuda or
        env['torch']['devices'][0]['name'] != torch.cuda.get_device_name(0)):
        raise ValueError('live runtime differs from environment record')
    payload, entry = load_accepted_layer(args.artifact_root, 0)
    original = {f: payload[args.module+'.'+f] for f in FIELDS}
    layout = {k: v.cuda() for k, v in convert_artifact(original).items()}
    dense = decode_layout(layout, torch.bfloat16)
    o, k = dense.shape
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    x = torch.randn(1, k, device='cuda', dtype=torch.bfloat16)
    outputs = {a: torch.empty(1, o, device='cuda', dtype=x.dtype) for a in ('dense','v1','v2','v3')}
    work = {a: torch.empty(workspace_shape(o,k,4), device='cuda') for a in ('v1','v2','v3')}
    functions = {'dense': lambda: torch.mm(x, dense.t(), out=outputs['dense'])}
    for arm in work:
        functions[arm] = lambda arm=arm: m1_out(x, layout, outputs[arm], work[arm],
                                               kernel=arm, groups_per_split=4)
    report = {'status':'running', 'diagnostic_only':True, 'module':args.module,
              'mode':args.mode, 'arm':args.arm, 'shape':[o,k], 'groups_per_split':4,
              'payload':entry, 'input_sha256':tensor_sha256(x), 'source_sha256':sources,
              'environment_sha256':sha256_file(args.environment),
              'runner_sha256':sha256_file(Path(__file__)),
              'scope':'layer 0 real weights, BF16 M=1; no full-model claim',
              'checks':{}}
    try:
        for fn in functions.values():
            fn()
        for arm in work:
            check = numerical_gate(outputs[arm], outputs['dense'])
            saved = outputs[arm].clone()
            functions[arm]()
            check['repeat_exact'] = bool(torch.equal(saved, outputs[arm]))
            report['checks'][arm] = check
            if not check['passed'] or not check['repeat_exact']:
                raise RuntimeError('numerical/repeat gate failed: '+arm)
        if args.mode == 'timing':
            graphs = {}
            for arm, fn in functions.items():
                for _ in range(20): fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): fn()
                graphs[arm] = graph
            report['timings'] = paired_cuda_timing({a:g.replay for a,g in graphs.items()})
            report['status'] = ('completed_pending_review' if all(t['stable'] for t in report['timings'].values())
                                else 'completed_unstable')
        else:
            fn = functions[args.arm]
            for _ in range(20): fn()
            torch.cuda.synchronize()
            # ncu --profile-from-start off: exclude load, reconstruction and JIT.
            torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range('m1_diagnostic_'+args.arm):
                    for _ in range(3): fn()
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            report['status'] = 'capture_completed_pending_review'
            report['profile_calls'] = 3
            report['timing_warning'] = 'No speed ratios from a profiler-instrumented process'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
