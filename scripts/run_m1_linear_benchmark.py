#!/usr/bin/env python3
"""Explicit M=1 Linear trial on frozen step400 payloads; never launches a block/model job."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from fluxbin_style.deployment import (KERNELS, FIELDS, convert_artifact, restore_artifact, decode_layout,
                                      conversion_record, workspace_shape, m1_out, load_extension)
from fluxbin_style.evaluation import sha256_file, tensor_sha256, atomic_json
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing
from fluxbin_style.factored_reference import REFERENCE, structural_reference, structural_gate

ROOT=Path(__file__).resolve().parents[1]
from fluxbin_style.deployment_artifacts import MANIFEST_SHA, load_accepted_layer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--groups-per-split',type=int,default=8)
    p.add_argument('--dtype',choices=('bf16','fp16'),default='bf16')
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--mode',choices=('eager','graph'),default='eager')
    p.add_argument('--kernel',choices=KERNELS,default='v1')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA required; no local fallback')
    env=json.loads(args.environment.read_text())
    if env['status']!='ready_for_gpu_trial':raise ValueError('environment preflight not ready')
    if env['torch']['version']!=torch.__version__ or env['torch']['cuda_runtime']!=torch.version.cuda:
        raise ValueError('environment torch/CUDA drift')
    if env['torch']['devices'][0]['name']!=torch.cuda.get_device_name(0):raise ValueError('GPU drift')
    source={str(f.relative_to(ROOT)):sha256_file(f) for f in
            sorted((ROOT/'src/fluxbin_style').rglob('*')) if f.suffix in ('.py','.cu')}
    if env['source_sha256']!=source:raise ValueError('source changed since environment capture')
    tensors,entry=load_accepted_layer(args.artifact_root,args.layer)
    torch.manual_seed(20260914);torch.cuda.manual_seed_all(20260914)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
    report={'status':'running','stage':'linear_m1','manifest_sha256':MANIFEST_SHA,'layer':args.layer,
            'payload':entry,'environment_sha256':sha256_file(args.environment),'source_sha256':source,
            'runner_sha256':sha256_file(Path(__file__)), 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'inputs':'seeded synthetic activations on real step400 weights',
            'baseline':'same-input dense BF16/FP16 matmul using decoded step400 weights',
            'timing_scope':'preallocated outputs; packed compute plus split reduction; v4 includes activation transform, v3 single split directly stores output; excludes conversion/JIT/load',
            'v1_comparison_policy':'diagnostic_only' if args.kernel in ('v3','v4') else 'exact',
            'numerical_reference':REFERENCE if args.kernel=='v4' else 'combined_weight_activation_dtype_v1',
            'next_stage':'not_launched','cells':[]}
    try:
        load_extension(args.kernel)
        if args.kernel!="v1":load_extension("v1")
        for name in QWEN3_LINEAR_MODULES:
            original={f:tensors[name+'.'+f] for f in FIELDS}
            layout=convert_artifact(original)
            restored=restore_artifact(layout)
            if any(not torch.equal(original[f],restored[f]) for f in FIELDS):
                raise ValueError('conversion changed payload')
            record=conversion_record(original,layout)
            layout={k:v.cuda() for k,v in layout.items()}
            dense=decode_layout(layout,dtype)
            o,k=dense.shape
            x=torch.randn(1,k,device='cuda',dtype=dtype)
            y=torch.empty(1,o,device='cuda',dtype=dtype);reference=torch.empty_like(y)
            workspace=torch.empty(workspace_shape(o,k,args.groups_per_split,kernel=args.kernel),device='cuda')
            y_v1=torch.empty_like(y) if args.kernel!='v1' else None
            workspace_v1=torch.empty(workspace_shape(o,k,args.groups_per_split),device='cuda') if args.kernel!='v1' else None
            def packed():return m1_out(x,layout,y,workspace,groups_per_split=args.groups_per_split,kernel=args.kernel)
            def baseline_v1():return m1_out(x,layout,y_v1,workspace_v1,groups_per_split=args.groups_per_split,kernel='v1')
            def baseline():return torch.mm(x,dense.t(),out=reference)
            cell={'module':name,'shape':[o,k],'conversion':record,'checks':[]}
            def check_output():
                legacy=numerical_gate(y,reference)
                if args.kernel!='v4':return legacy
                gate=structural_gate(y,structural_reference(x,layout))
                gate['legacy_dense_bf16']=legacy
                return gate
            # Multiple inputs plus a cancellation-sensitive alternating-sign probe.
            for seed in (20260914,20260915,20260916):
                torch.cuda.manual_seed_all(seed);x.normal_();baseline();packed()
                saved=y.clone();workspace.fill_(float('nan'));packed()
                gate=check_output();gate['repeat_exact']=bool(torch.equal(saved,y));gate['seed']=seed
                if args.kernel!='v1':
                    baseline_v1();gate['v1_exact']=bool(torch.equal(y,y_v1))
                cell['checks'].append(gate)
            x.copy_((torch.arange(k,device='cuda')%2*2-1).to(dtype));baseline();packed()
            gate=check_output()
            if args.kernel!='v1':
                baseline_v1();gate['v1_exact']=bool(torch.equal(y,y_v1))
            cell['checks'].append(gate)
            cell['correctness_passed']=all(c['passed'] and c.get('repeat_exact',True) and (args.kernel in ('v3','v4') or c.get('v1_exact',True)) for c in cell['checks'])
            report['cells'].append(cell)
            if not cell['correctness_passed']:raise RuntimeError(f'numerical gate failed: {name}')
            # Fixed random timing input, not the alternating diagnostic input.
            torch.cuda.manual_seed_all(20260914);x.normal_();baseline();packed()
            cell['input_sha256']=tensor_sha256(x)
            functions={'dense':baseline,'packed':packed}
            if args.kernel!='v1':functions['v1']=baseline_v1
            graphs=[]
            if args.mode=='graph':
                for fn in functions.values():
                    for _ in range(3):fn()
                torch.cuda.synchronize()
                for fn in functions.values():
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):fn()
                    graphs.append(graph)
                functions={key:graph.replay for key,graph in zip(functions,graphs)}
                for fn in functions.values():fn()
                if not check_output()['passed']:raise RuntimeError('graph replay numerical gate failed')
                if args.kernel not in ('v1','v3','v4') and not torch.equal(y,y_v1):raise RuntimeError('graph v1/v2 equality failed')
            cell['timings']=paired_cuda_timing(functions,warmup=args.warmup,repeats=args.repeats,rounds=args.rounds)
            stable=all(t['stable'] for t in cell['timings'].values())
            cell['speedup']=cell['timings']['dense']['median_us']/cell['timings']['packed']['median_us'] if stable else None
            cell['timing_stable']=stable
            if args.kernel!='v1':
                cell['speedup_vs_v1']=cell['timings']['v1']['median_us']/cell['timings']['packed']['median_us'] if stable else None
            atomic_json(args.output,report)
            # Release graph references before the next cell to avoid retaining weights.
            del graphs,functions,layout,dense,workspace,workspace_v1,y_v1
        report['status']='completed_pending_review' if all(c['timing_stable'] for c in report['cells']) else 'completed_unstable'
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        atomic_json(args.output,report)
    print(report['status'])


if __name__=='__main__':main()
