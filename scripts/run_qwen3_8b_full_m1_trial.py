#!/usr/bin/env python3
"""Explicit full Qwen3-8B cached M=1 trial after matching block gate. No vLLM."""
from __future__ import annotations
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from fluxbin_style.evaluation import atomic_json,sha256_file,tensor_sha256,materialize_hybrid_s8_weight
from fluxbin_style.deployment_artifacts import MANIFEST_SHA,load_accepted_layer,replace_model_linears
from fluxbin_style.deployment import FIELDS,load_extension
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.full_model_trial import decode_trace,compare_trace,validate_routes
from fluxbin_style.trial_gates import validate_block_gate

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/acceleration/qwen3_8b_full_m1_v1.json'


def summary(traces):
    out={}
    for key in ('prefill_wall_ms','decode_wall_ms','prefill_device_ms','decode_device_ms'):
        values=[t[key] for t in traces];median=statistics.median(values)
        out[key]={'samples':values,'median':median,'relative_range':(max(values)-min(values))/median}
    return out


def small_trace(trace):
    return {**{k:v for k,v in trace.items() if k not in ('logits','predictions','fed_tokens')},
            'predictions':trace['predictions'].tolist(),'fed_tokens':trace['fed_tokens'].tolist(),
            'logits_sha256':tensor_sha256(trace['logits'])}


def must_abort(row, numerical_policy):
    check=row['correctness']
    if not row.get('coverage_passed',True) or not check.get('fed_tokens_equal',True):
        return True
    if check.get('logits',{}).get('reason')=='nonfinite output':
        return True
    return not check['passed'] and numerical_policy=='strict'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot-root','artifact-root','environment','block-result','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--numerical-policy',choices=('strict','report-only'),default='strict',
                   help='report-only records finite numerical differences without stopping performance measurement')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA required; no CPU/MPS substitution')
    cfg=json.loads(PROTOCOL.read_text());env=json.loads(args.environment.read_text())
    sources={str(f.relative_to(ROOT)):sha256_file(f) for f in sorted((ROOT/'src/fluxbin_style').rglob('*')) if f.suffix in ('.py','.cu')}
    if env['status']!='ready_for_gpu_trial' or env['source_sha256']!=sources:
        raise ValueError('capture a fresh matching environment/source record')
    if (env['torch']['version']!=torch.__version__ or env['torch']['cuda_runtime']!=torch.version.cuda
        or env['torch']['devices'][0]['name']!=torch.cuda.get_device_name(0)):
        raise ValueError('runtime differs from environment record')
    block=json.loads(args.block_result.read_text())
    kernel,gps=validate_block_gate(block,source_sha256=sources,environment_sha256=sha256_file(args.environment))
    pinned=json.loads((ROOT/'configs/evaluation/qwen3_8b_wikitext2_distilled_step400_v1.json').read_text())
    if args.snapshot_root.name!=pinned['model']['revision']:raise ValueError('snapshot revision drift')
    for name,digest in pinned['model_preflight_files'].items():
        if sha256_file(args.snapshot_root/name)!=digest:raise ValueError(f'snapshot hash drift: {name}')
    # Validate all immutable payloads before any model is executed.
    payload_records=[]
    for layer in range(36):
        _,entry=load_accepted_layer(args.artifact_root,layer);payload_records.append(entry)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(args.snapshot_root,local_files_only=True)
    prompts=[tokenizer(text,return_tensors='pt').input_ids for text in cfg['prompts']]
    if any(not 2<=x.shape[1]<=cfg['max_prompt_tokens'] for x in prompts):
        raise ValueError('prompt length outside protocol; do not silently truncate')
    load_extension(kernel)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    report={'status':'running','stage':'full_model_cached_m1','protocol':cfg,
            'protocol_sha256':sha256_file(PROTOCOL),'source_sha256':sources,
            'runner_sha256':sha256_file(Path(__file__)),
            'block_result_sha256':sha256_file(args.block_result),'environment_sha256':sha256_file(args.environment),
            'manifest_sha256':MANIFEST_SHA,'payloads':payload_records,'kernel':kernel,'groups_per_split':gps,
            'prompt_token_sha256':[tensor_sha256(x) for x in prompts],
            'scope':'HF eager, dynamic KV cache, batch1; identical forced continuation from decoded step400',
            'prefill_policy':'packed arm uses explicit on-demand dense reconstruction; included in prefill time',
            'load_policy':'load pinned dense BF16 snapshot, then replace; not direct packed loading',
            'numerical_policy':args.numerical_policy,'all_numerical_checks_passed':True,
            'primary_baseline':'original_bf16','next_stage':'not_launched','arms':{}}
    references={};all_stable=True
    try:
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            model=AutoModelForCausalLM.from_pretrained(args.snapshot_root,local_files_only=True,
                  dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
            torch.cuda.synchronize();load_seconds=time.perf_counter()-start
            start=time.perf_counter();coverage=None
            if arm=='decoded_step400':
                for layer,decoder in enumerate(model.model.layers):
                    payload,_=load_accepted_layer(args.artifact_root,layer)
                    for name in QWEN3_LINEAR_MODULES:
                        weight=materialize_hybrid_s8_weight(**{f:payload[name+'.'+f] for f in FIELDS},
                               group_size=128,columns_per_group=8,device='cuda',output_dtype=torch.bfloat16)
                        decoder.get_submodule(name).weight.copy_(weight)
                    del payload,weight
                del decoder
            elif arm=='packed_step400':
                coverage=replace_model_linears(model,args.artifact_root,allow_prefill_fallback=True,
                                              groups_per_split=gps,kernel=kernel)
                if coverage['linear_count']!=252:raise RuntimeError('incomplete packed coverage')
            torch.cuda.synchronize()
            arm_record={'snapshot_load_seconds':load_seconds,'conversion_seconds':time.perf_counter()-start,
                        'load_and_conversion_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'resident_allocated_bytes':torch.cuda.memory_allocated(),'coverage':coverage,'prompts':[]}
            report['arms'][arm]=arm_record
            for prompt_index,input_ids in enumerate(prompts):
                ids=input_ids.cuda();reference=references.get(prompt_index)
                forced=None if reference is None else reference['fed_tokens']
                for _ in range(cfg['warmup']):
                    decode_trace(model,ids,steps=cfg['decode_steps'],forced_tokens=forced)
                torch.cuda.reset_peak_memory_stats();traces=[];rows=[]
                for repeat in range(cfg['repeats']):
                    trace=decode_trace(model,ids,steps=cfg['decode_steps'],forced_tokens=forced)
                    if arm=='decoded_step400' and repeat==0:
                        references[prompt_index]=trace;reference=trace;forced=trace['fed_tokens']
                    row=small_trace(trace);rows.append(row);traces.append(trace)
                    if arm!='original_bf16':
                        row['correctness']=compare_trace(trace,reference,logprob_tolerance=cfg['logprob_max_abs_tolerance'])
                        if arm=='packed_step400':
                            row['coverage_passed']=validate_routes(trace,expected_linears=252,steps=cfg['decode_steps'])
                        report['all_numerical_checks_passed'] &= row['correctness']['passed']
                        if must_abort(row,args.numerical_policy):
                            arm_record['prompts'].append({'prompt_index':prompt_index,'traces':rows})
                            raise RuntimeError(f'full-model correctness/coverage failed: {arm}:{prompt_index}:{repeat}')
                timings=summary(traces)
                stable=all(timings[k]['relative_range']<=cfg['max_relative_timing_range']
                           for k in ('decode_wall_ms','decode_device_ms'))
                all_stable=all_stable and stable
                arm_record['prompts'].append({'prompt_index':prompt_index,'traces':rows,'timings':timings,
                         'decode_timing_stable':stable,'runtime_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'decode_tokens_per_second':cfg['decode_steps']*1000/timings['decode_wall_ms']['median']})
                atomic_json(args.output,report)
            del model,trace,traces;gc.collect();torch.cuda.empty_cache()
        comparisons=[]
        for i in range(len(prompts)):
            rows={arm:report['arms'][arm]['prompts'][i] for arm in cfg['arms']}
            stable=all(r['decode_timing_stable'] for r in rows.values())
            candidate=rows['packed_step400']['timings']['decode_wall_ms']['median']
            comparisons.append({'prompt_index':i,'stable':stable,
                'decode_wall_speedup_vs_decoded':rows['decoded_step400']['timings']['decode_wall_ms']['median']/candidate if stable else None,
                'decode_wall_speedup_vs_original':rows['original_bf16']['timings']['decode_wall_ms']['median']/candidate if stable else None})
        report['comparisons']=comparisons
        report['all_timings_stable']=all_stable
        report['status']=('completed_unstable' if not all_stable else
                          'completed_pending_review' if report['all_numerical_checks_passed'] else
                          'completed_with_numerical_differences')
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:atomic_json(args.output,report)
    print(report['status'])


if __name__=='__main__':main()
