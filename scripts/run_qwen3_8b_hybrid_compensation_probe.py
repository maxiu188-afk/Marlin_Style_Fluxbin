#!/usr/bin/env python3
"""Same-input hybrid compensation probe. No pure, full-model run or distillation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from fluxbin_style import (InputHessianAccumulator, TwoBaseRankOneOptimizationConfig,
    atomic_json, invert_hessian, materialize_hybrid_s8_weight, pack_two_bases,
    quantize_hybrid_two_base_obq, sha256_file, tensor_sha256)
from fluxbin_style.hybrid_conditioned import quantize_hybrid_conditioned_v1
from run_qwen3_8b_single_linear_hessian_obq_s8 import reconstruction_metrics
from run_qwen3_two_base_rank1_s8_ppl import validate_runtime

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiments/qwen3_8b_hybrid_compensation_probe_v1.json'
PCIE_CONFIG=CONFIG.with_name('qwen3_8b_hybrid_compensation_probe_v1_a100_pcie.json')


def validate_probe_config(config):
    if not any(config==json.loads(path.read_text()) for path in (CONFIG,PCIE_CONFIG)):
        raise ValueError('unknown or modified frozen probe config')


def load_probe_tokens(path,config):
    stored=load_file(path)
    if set(stored)!={'token_ids'}:raise ValueError('C4 calibration must contain token_ids only')
    tokens=stored['token_ids']
    if list(tokens.shape)!=config['token_shape'] or tokens.dtype!=torch.int32 or torch.any(tokens<0):
        raise ValueError('token contract drift')
    return tokens


def payload_of(result):
    d=result.decomposition; g=d.global_decomposition; s=d.refinement_decomposition
    return {k:v.detach().cpu().contiguous() for k,v in {
        'global_sign_codes':pack_two_bases(g.bases), 'global_row_scales':g.row_scales,
        'global_column_scales':g.column_scales, 'refinement_indices':d.selected_indices.to(torch.int16),
        'refinement_sign_codes':pack_two_bases(s.bases), 'refinement_row_scales':s.row_scales,
        'refinement_column_scales':s.column_scales}.items()}


class CaptureDone(Exception):
    pass


def walk_prefix(model,tokens,groups,callback,device):
    """Capture identical original BF16 inputs; abort before last target computes."""
    handles=[]; counts={name:0 for name in groups}; digests={name:hashlib.sha256() for name in groups}
    last=list(groups)[-1]
    def hook(name):
        def collect(module,inputs):
            x=inputs[0].detach()
            counts[name]+=x.numel()//x.shape[-1]
            digests[name].update(tensor_sha256(x).encode())
            callback(name,x)
            if name==last:raise CaptureDone()
        return collect
    try:
        for name in groups:handles.append(model.get_submodule(name).register_forward_pre_hook(hook(name)))
        with torch.inference_mode():
            for i,row in enumerate(tokens):
                try:model(input_ids=row[None].to(device=device,dtype=torch.long),use_cache=False)
                except CaptureDone:pass
                else:raise RuntimeError('prefix stop hook did not fire')
                if (i+1)%32==0:print(f'INPUT_PASS={i+1}/{len(tokens)}',flush=True)
    finally:
        for h in handles:h.remove()
    return {'rows':counts,'ordered_input_tensor_hashes':{k:v.hexdigest() for k,v in digests.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--snapshot-root',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Run GPU fits; default only validates inputs/runtime')
    args=p.parse_args();c=json.loads(args.config.read_text());validate_probe_config(c)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.snapshot_root.name!=c['model']['revision']:raise ValueError('model revision mismatch')
    for name,digest in c['model_preflight_files'].items():
        if sha256_file(args.snapshot_root/name)!=digest:raise ValueError(f'model file drift: {name}')
    for filename,key in [('manifest.json','calibration_manifest_sha256'),('tokens.safetensors','calibration_tokens_sha256')]:
        if sha256_file(args.calibration_dir/filename)!=c[key]:raise ValueError('calibration drift')
    tokens=load_probe_tokens(args.calibration_dir/'tokens.safetensors',c)
    device=validate_runtime(c)
    print('PROBE_PREFLIGHT=passed',flush=True)
    if not args.execute:return
    args.output_dir.mkdir(parents=True)
    sources=list((ROOT/'src/fluxbin_style').glob('*.py'))+[CONFIG,PCIE_CONFIG,Path(__file__),ROOT/'scripts/run_qwen3_8b_single_linear_hessian_obq_s8.py',ROOT/'scripts/run_qwen3_two_base_rank1_s8_ppl.py']
    source_hashes={str(x.relative_to(ROOT)):sha256_file(x) for x in sources}
    started=time.monotonic()
    atomic_json(args.output_dir/'provenance.json',{'config':c,'config_sha256':sha256_file(args.config),'source_files_sha256':source_hashes})
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(args.snapshot_root,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    model.config.use_cache=False
    groups=c['capture_groups']
    # JSON writers may sort keys: explicit target order must put the terminal hook last.
    groups={name:groups[name] for name in ('model.layers.1.mlp.gate_proj','model.layers.6.mlp.down_proj')}
    accum={name:InputHessianAccumulator(model.get_submodule(name).in_features,device=device) for name in groups}
    trace=walk_prefix(model,tokens,groups,lambda n,x:accum[n].add(x),device)
    if any(n!=256*2048 for n in trace['rows'].values()):raise ValueError('capture count drift')
    records={}; reconstructions={}
    for group,targets in groups.items():
        h=accum.pop(group).value(); inverse=invert_hessian(h,damp_percent=c['damp_percent']).inverse
        for target in targets:
            print(f'FIT_TARGET={target}',flush=True)
            w=model.get_submodule(target).weight.detach(); fixed=None
            records[target]={'target_bf16_sha256':tensor_sha256(w),'hessian_sha256':tensor_sha256(h),'arms':{}}
            reconstructions[target]={}
            for arm in c['arms']:
                kwargs=dict(group_size=c['group_size'],columns_per_group=c['columns_per_group'],global_config=TwoBaseRankOneOptimizationConfig(**c['global_solver']),refinement_config=TwoBaseRankOneOptimizationConfig(**c['refinement_solver']))
                torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                if arm=='legacy_hybrid':
                    fit=quantize_hybrid_two_base_obq(w.float(),inverse,**kwargs)
                    fixed=fit.decomposition.selected_indices.clone()
                else:fit=quantize_hybrid_conditioned_v1(w.float(),inverse,fixed_indices=fixed,**kwargs)
                tensors=payload_of(fit)
                path=args.output_dir/f'{target}.{arm}.safetensors';save_file(tensors,path)
                stored=load_file(path)
                if any(not torch.equal(v,stored[k]) for k,v in tensors.items()):raise ValueError('payload round trip failed')
                if not torch.equal(stored['refinement_indices'].long().to(device),fixed):raise ValueError('selected columns changed')
                q=materialize_hybrid_s8_weight(**stored,group_size=c['group_size'],columns_per_group=c['columns_per_group'],device=device,output_dtype=torch.bfloat16)
                rec={'metrics':reconstruction_metrics(w.float(),q,h),'payload_sha256':sha256_file(path),'indices_sha256':tensor_sha256(stored['refinement_indices']),'elapsed_seconds':time.monotonic()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                records[target]['arms'][arm]=rec;reconstructions[target][arm]=q.cpu()
                del fit,q,tensors,stored
            del fixed
        del h,inverse
        torch.cuda.empty_cache()
    atomic_json(args.output_dir/'fits.json',{'input_trace':trace,'targets':records})
    totals={target:{arm:[] for arm in c['arms']} for target in records}
    errors={target:{arm:(model.get_submodule(target).weight.detach().float()-q.to(device).float()).T.contiguous() for arm,q in variants.items()} for target,variants in reconstructions.items()}
    def direct(group,x):
        for target in groups[group]:
            for arm,error in errors[target].items():
                for chunk in x.reshape(-1,x.shape[-1]).split(128):
                    totals[target][arm].append(float((chunk.float()@error).square().sum(dtype=torch.float64)))
    replay=walk_prefix(model,tokens,groups,direct,device)
    if trace!=replay:raise ValueError('BF16 input replay drifted')
    valid=True;decisions={}
    for target in records:
        for arm in c['arms']:
            direct_loss=math.fsum(totals[target][arm])/(256*2048)
            rec=records[target]['arms'][arm]; expected=rec['metrics']['calibration_total_output_squared_error']
            agreement=math.isfinite(direct_loss) and math.isclose(direct_loss,expected,rel_tol=c['output_loss_relative_tolerance'],abs_tol=c['output_loss_absolute_tolerance'])
            rec.update(direct_mean_token_output_squared_error=direct_loss,hessian_direct_agreement=agreement)
            valid=valid and agreement
        old=records[target]['arms']['legacy_hybrid']['direct_mean_token_output_squared_error']
        new=records[target]['arms']['conditioned_fixed_indices']['direct_mean_token_output_squared_error']
        decisions[target]={'output_loss_reduced':new<old,'relative_output_loss_change':(new-old)/old if old else None}
    if source_hashes!={str(x.relative_to(ROOT)):sha256_file(x) for x in sources}:raise ValueError('source changed during run')
    atomic_json(args.output_dir/'result.json',{'status':'completed_pending_review' if valid else 'failed_metric_agreement','config_sha256':sha256_file(args.config),'input_policy':c['input_policy'],'input_trace':trace,'source_files_sha256':source_hashes,'targets':records,'comparison':decisions,'elapsed_seconds':time.monotonic()-started,'full_model_auto_launch':False,'distillation_auto_launch':False})
    print('PROBE_COMPLETE; manual review required',flush=True)
    if not valid:raise SystemExit(1)


if __name__=='__main__':main()
