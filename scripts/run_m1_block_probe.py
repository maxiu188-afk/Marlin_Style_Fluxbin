#!/usr/bin/env python3
"""Single real Qwen3 block, M=1, empty KV cache: bounded correctness/timing probe.

Run AFTER reviewing Linear results. This probe does not measure long-context
cached decode, full-model inference or serving. No next stage is launched.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from safetensors import safe_open
from fluxbin_style.deployment_artifacts import load_accepted_layer, MANIFEST_SHA
from fluxbin_style.deployment import FIELDS, PackedHybridLinear, replace_block_linears
from fluxbin_style.evaluation import materialize_hybrid_s8_weight, sha256_file, atomic_json
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.acceleration_checks import numerical_gate, paired_cuda_timing

ROOT=Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot-root','artifact-root','linear-result','environment','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA required')
    prior=json.loads(args.linear_result.read_text())
    if prior.get('status')!='completed_pending_review' or prior.get('manifest_sha256')!=MANIFEST_SHA or prior.get('layer')!=args.layer:
        raise ValueError('matching completed stable Linear trial required; review it before this explicit launch')
    kernel=prior['settings'].get('kernel','v1')
    if prior['settings']['dtype']!='bf16':raise ValueError('BF16 Linear trial required for BF16 block')
    if {c['module'] for c in prior['cells']}!=set(QWEN3_LINEAR_MODULES) or not all(
        c['correctness_passed'] and c['timing_stable'] for c in prior['cells']):
        raise ValueError('all seven stable, correct Linear cells required')
    groups_per_split=prior['settings']['groups_per_split']
    env=json.loads(args.environment.read_text())
    sources={str(f.relative_to(ROOT)):sha256_file(f) for f in sorted((ROOT/'src/fluxbin_style').rglob('*')) if f.suffix in ('.py','.cu')}
    if env['status']!='ready_for_gpu_trial' or env['source_sha256']!=sources or prior['source_sha256']!=sources:
        raise ValueError('environment/Linear source drift')
    if prior['environment_sha256']!=sha256_file(args.environment):raise ValueError('Linear environment drift')
    if env['torch']['version']!=torch.__version__ or env['torch']['cuda_runtime']!=torch.version.cuda or env['torch']['devices'][0]['name']!=torch.cuda.get_device_name(0):
        raise ValueError('live runtime drift')
    config_record=json.loads((ROOT/'configs/evaluation/qwen3_8b_wikitext2_distilled_step400_v1.json').read_text())
    if args.snapshot_root.name!=config_record['model']['revision']:raise ValueError('snapshot revision drift')
    for name,digest in config_record['model_preflight_files'].items():
        if sha256_file(args.snapshot_root/name)!=digest:raise ValueError(f'snapshot drift: {name}')
    payload,entry=load_accepted_layer(args.artifact_root,args.layer)
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
    config=AutoConfig.from_pretrained(args.snapshot_root,local_files_only=True)
    config._attn_implementation='sdpa'
    prefix=f'model.layers.{args.layer}.'
    index=json.loads((args.snapshot_root/'model.safetensors.index.json').read_text())['weight_map']
    state={}
    for shard in sorted({s for n,s in index.items() if n.startswith(prefix)}):
        with safe_open(args.snapshot_root/shard,framework='pt',device='cpu') as f:
            for name in f.keys():
                if name.startswith(prefix):state[name[len(prefix):]]=f.get_tensor(name)
    with torch.device('meta'):block=Qwen3DecoderLayer(config,args.layer)
    block.load_state_dict(state,strict=True,assign=True)
    block=block.cuda().eval()
    original=copy.deepcopy(block)
    for name in QWEN3_LINEAR_MODULES:
        tensors={field:payload[name+'.'+field] for field in FIELDS}
        w=materialize_hybrid_s8_weight(**tensors,group_size=128,columns_per_group=8,
                                      device='cuda',output_dtype=torch.bfloat16)
        block.get_submodule(name).weight.copy_(w)
    packed=copy.deepcopy(block)
    replace_block_linears(packed,payload,groups_per_split=groups_per_split,kernel=kernel)
    rotary=Qwen3RotaryEmbedding(config,device='cuda')
    torch.manual_seed(20260914);torch.cuda.manual_seed_all(20260914)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    x=torch.randn(1,1,config.hidden_size,device='cuda',dtype=torch.bfloat16)
    position=torch.zeros(1,1,device='cuda',dtype=torch.long)
    def call(module):
        # Include RoPE generation, norms, attention, residuals and all seven Linears.
        return module(x,position_ids=position,position_embeddings=rotary(x,position),use_cache=False)
    record={'status':'running','stage':'single_block_empty_cache_m1','layer':args.layer,'payload':entry,
            'manifest_sha256':MANIFEST_SHA,'environment_sha256':sha256_file(args.environment),
            'linear_result_sha256':sha256_file(args.linear_result),'source_sha256':sources,
            'runner_sha256':sha256_file(Path(__file__)), 'next_stage':'not_launched',
            'scope':'synthetic hidden inputs, real weights; sequence=1, no KV prefix; eager only',
            'groups_per_split':groups_per_split,'kernel':kernel,
            'timing_settings':{'warmup':20,'repeats':100,'rounds':7},'checks':[]}
    try:
        for _ in range(3):
            x.normal_();ref=call(block);actual=call(packed)
            check=numerical_gate(actual,ref)
            check['repeat_exact']=bool(torch.equal(actual,call(packed)))
            record['checks'].append(check)
        if not all(c['passed'] and c['repeat_exact'] for c in record['checks']):
            raise RuntimeError('block correctness failed; do not promote to full model')
        if any(m.last_route!='packed_m1' for m in packed.modules() if isinstance(m,PackedHybridLinear)):
            raise RuntimeError('unexpected fallback')
        record['kernel_coverage']=7
        record['timings']=paired_cuda_timing({'original_bf16':lambda:call(original),
                                             'decoded_step400_bf16':lambda:call(block),
                                             'packed_step400':lambda:call(packed)})
        stable=all(t['stable'] for t in record['timings'].values())
        record['status']='completed_pending_review' if stable else 'completed_unstable'
    except Exception as exc:
        record['status']='failed';record['error']=f'{type(exc).__name__}: {exc}';raise
    finally:atomic_json(args.output,record)
    print(record['status'])


if __name__=='__main__':main()
