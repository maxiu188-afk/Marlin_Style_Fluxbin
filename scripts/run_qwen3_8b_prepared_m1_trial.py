#!/usr/bin/env python3
"""Versioned static-KV, prepared eager and full-sequence Graph comparison. CUDA only."""
from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fluxbin_style.deployment import FIELDS, load_extension
from fluxbin_style.deployment_artifacts import MANIFEST_SHA, load_accepted_layer, replace_model_linears
from fluxbin_style.evaluation import atomic_json, sha256_file, tensor_sha256, materialize_hybrid_s8_weight
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.full_model_trial import decode_trace, compare_trace, validate_routes
from fluxbin_style.static_decode import StaticDecodeSession, prepared_linears

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/acceleration/qwen3_8b_full_m1_v2.json'


def statistics_row(samples, threshold):
    if not samples or not all(math.isfinite(x) and x>0 for x in samples):
        raise ValueError('invalid timing samples')
    median=statistics.median(samples)
    span=(max(samples)-min(samples))/median
    return dict(samples=samples,median=median,relative_range=span,stable=span<=threshold)


def exact_trace(actual, expected):
    if actual['cache_length']!=expected['cache_length']:
        raise RuntimeError('cache length drift')
    for key in ('logits','predictions','fed_tokens'):
        if not torch.equal(actual[key],expected[key]):
            raise RuntimeError('repeat/Graph equivalence failed: '+key)


def require_routes(routes, steps):
    if len(routes)!=252 or any(r!={'dense_fallback':0,'packed_m1':steps} for r in routes.values()):
        raise RuntimeError('prepared decode route coverage failed')


def compact(trace):
    return dict(logits_sha256=tensor_sha256(trace['logits']),
                predictions=trace['predictions'].tolist(),fed_tokens=trace['fed_tokens'].tolist(),
                cache_length=trace['cache_length'],routes=trace['routes'])


def load_model(snapshot, artifacts, arm, kernel, gps):
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
          dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    torch.cuda.synchronize()
    coverage=None
    if arm=='decoded_step400':
        for layer,decoder in enumerate(model.model.layers):
            payload,_=load_accepted_layer(artifacts,layer)
            for name in QWEN3_LINEAR_MODULES:
                weight=materialize_hybrid_s8_weight(**{f:payload[name+'.'+f] for f in FIELDS},
                       group_size=128,columns_per_group=8,device='cuda',output_dtype=torch.bfloat16)
                decoder.get_submodule(name).weight.copy_(weight)
            del payload,weight
        del decoder
    elif arm=='packed_step400':
        coverage=replace_model_linears(model,artifacts,allow_prefill_fallback=True,
                                      groups_per_split=gps,kernel=kernel)
        if coverage['linear_count']!=252:raise RuntimeError('incomplete packed coverage')
    torch.cuda.synchronize()
    return model,coverage


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot-root','artifact-root','environment','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA required; no CPU/MPS substitution')
    cfg=json.loads(PROTOCOL.read_text());env=json.loads(args.environment.read_text())
    sources={str(f.relative_to(ROOT)):sha256_file(f) for f in sorted((ROOT/'src/fluxbin_style').rglob('*'))
             if f.suffix in ('.py','.cu','.cuh')}
    if env['status']!='ready_for_gpu_trial' or env['source_sha256']!=sources:
        raise ValueError('capture a fresh matching environment/source record')
    if (env['torch']['version']!=torch.__version__ or env['torch']['cuda_runtime']!=torch.version.cuda
        or env['torch']['devices'][0]['name']!=torch.cuda.get_device_name(0)):
        raise ValueError('runtime differs from environment record')
    pinned=json.loads((ROOT/'configs/evaluation/qwen3_8b_wikitext2_distilled_step400_v1.json').read_text())
    if args.snapshot_root.name!=pinned['model']['revision']:raise ValueError('snapshot revision drift')
    for name,digest in pinned['model_preflight_files'].items():
        if sha256_file(args.snapshot_root/name)!=digest:raise ValueError('snapshot hash drift: '+name)
    payloads=[load_accepted_layer(args.artifact_root,i)[1] for i in range(36)]
    tokenizer=AutoTokenizer.from_pretrained(args.snapshot_root,local_files_only=True)
    prompts=[tokenizer(text,return_tensors='pt').input_ids.cuda() for text in cfg['prompts']]
    if any(not 2<=x.shape[1]<=cfg['max_prompt_tokens'] for x in prompts):raise ValueError('prompt length drift')
    kernel,gps=cfg['kernel'],cfg['groups_per_split'];load_extension(kernel)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    report=dict(status='running',stage='prepared_static_full_model_m1',protocol=cfg,
        protocol_sha256=sha256_file(PROTOCOL),runner_sha256=sha256_file(Path(__file__)),
        environment_sha256=sha256_file(args.environment),source_sha256=sources,
        manifest_sha256=MANIFEST_SHA,payloads=payloads,kernel=kernel,groups_per_split=gps,
        prompt_token_sha256=[tensor_sha256(p) for p in prompts],
        primary_baseline='original_bf16',numerical_policy='cross-weight and dynamic/static attention report-only; same-static-cache wrapper and Graph exact',
        scope='batch1 real-prefix static KV, 32 fixed continuation tokens, embedding/all36blocks/LM head/argmax',
        excluded_from_decode='load, conversion, prefill, KV reset, graph capture, audits, CPU output copies',
        residency_policy='all three models and both prompt caches resident, interleaved arm order',
        interpretation='v2 protocol; do not attribute v1-to-v2 differences solely to kernel speed',
        arms={},execution_order=[],next_stage='not_launched')
    try:
        models={};sessions={};dynamic={};checked_static={};audits={};references={}
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            start=time.perf_counter();before=torch.cuda.memory_allocated()
            model,coverage=load_model(args.snapshot_root,args.artifact_root,arm,kernel,gps)
            models[arm]=model
            record=dict(load_conversion_seconds=time.perf_counter()-start,coverage=coverage,
                        resident_increment_bytes=torch.cuda.memory_allocated()-before,prompts=[])
            report['arms'][arm]=record
            for i,ids in enumerate(prompts):
                ref=references.get(i)
                trace=decode_trace(model,ids,steps=cfg['decode_steps'],
                                   forced_tokens=None if ref is None else ref['fed_tokens'])
                if not torch.isfinite(trace['logits']).all():raise RuntimeError('nonfinite dynamic oracle')
                if arm=='decoded_step400':references[i]=trace
                if arm=='packed_step400' and not validate_routes(trace,expected_linears=252,steps=cfg['decode_steps']):
                    raise RuntimeError('dynamic packed coverage failed')
                dynamic[arm,i]=trace
                start=time.perf_counter()
                sessions[arm,i]=StaticDecodeSession(model,ids,references[i]['fed_tokens'])
                torch.cuda.synchronize()
                checked_static[arm,i]=sessions[arm,i].audit()
                record['prompts'].append(dict(prompt_index=i,dynamic_audit=compact(trace),timings={},
                    static_prefix_setup_seconds=time.perf_counter()-start))
        with ExitStack() as stack:
            for model in models.values():stack.enter_context(prepared_linears(model,torch.bfloat16))
            for (arm,i),session in sessions.items():
                audit=session.audit();audits[arm,i]=audit
                check=compare_trace(audit,dynamic[arm,i],logprob_tolerance=cfg['logprob_max_abs_tolerance'])
                # Static masks can select different BF16 attention arithmetic from
                # dynamic KV. Isolate binding correctness against the same cache
                # and mask implementation, with the original checked wrapper.
                exact_trace(audit,checked_static[arm,i])
                if not check['fed_tokens_equal']:raise RuntimeError('dynamic/static fed-token drift')
                if arm=='packed_step400':require_routes(audit['routes'],cfg['decode_steps'])
                graph_trace=session.capture();exact_trace(graph_trace,audit)
                if arm=='packed_step400':require_routes(session.capture_routes,cfg['decode_steps'])
                report['arms'][arm]['prompts'][i].update(prepared_audit=compact(audit),
                    dynamic_static_check=check,checked_static_audit=compact(checked_static[arm,i]),
                    prepared_wrapper_exact=True,graph_exact=True,capture_routes=session.capture_routes)
            report['all_models_caches_graphs_allocated_bytes']=torch.cuda.memory_allocated()
            report['setup_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            # Warm the complete workload, not a single token or a single Linear.
            for _ in range(cfg['warmup']):
                for session in sessions.values():
                    for mode in cfg['modes']:session.measure(mode)
            samples={(arm,i,mode,key):[] for arm,i in sessions for mode in cfg['modes']
                     for key in ('wall_ms','device_ms')}
            for repeat in range(cfg['repeats']):
                arms=cfg['arms'][repeat%3:]+cfg['arms'][:repeat%3]
                if repeat%2:arms=list(reversed(arms))
                modes=cfg['modes'] if repeat%2==0 else list(reversed(cfg['modes']))
                for i in range(len(prompts)):
                    for mode in modes:
                        for arm in arms:
                            timing,trace=sessions[arm,i].measure(mode)
                            exact_trace(trace,audits[arm,i])
                            report['execution_order'].append(dict(repeat=repeat,prompt=i,mode=mode,arm=arm))
                            for key,value in timing.items():samples[arm,i,mode,key].append(value)
                # Persist completed samples even if a later round fails.
                for arm,i in sessions:
                    report['arms'][arm]['prompts'][i]['timings']={mode:{key:statistics_row(
                        samples[arm,i,mode,key],cfg['max_relative_timing_range']) for key in ('wall_ms','device_ms')}
                        for mode in cfg['modes']}
                atomic_json(args.output,report)
        comparisons=[]
        for i in range(len(prompts)):
            cross=compare_trace(audits['packed_step400',i],audits['decoded_step400',i],
                                logprob_tolerance=cfg['logprob_max_abs_tolerance'])
            for mode in cfg['modes']:
                rows={arm:report['arms'][arm]['prompts'][i]['timings'][mode] for arm in cfg['arms']}
                stable=all(v['stable'] for row in rows.values() for v in row.values())
                packed=rows['packed_step400']['wall_ms']['median']
                comparisons.append(dict(prompt_index=i,mode=mode,stable=stable,
                    speedup_vs_original=rows['original_bf16']['wall_ms']['median']/packed if stable else None,
                    speedup_vs_decoded=rows['decoded_step400']['wall_ms']['median']/packed if stable else None,
                    cross_weight_check=cross))
        report['comparisons']=comparisons
        report['all_timings_stable']=all(c['stable'] for c in comparisons)
        report['all_cross_weight_checks_passed']=all(c['cross_weight_check']['passed'] for c in comparisons)
        report['all_dynamic_static_checks_passed']=all(
            p['dynamic_static_check']['passed'] for a in report['arms'].values() for p in a['prompts'])
        report['all_numerical_checks_passed']=(report['all_cross_weight_checks_passed'] and
                                              report['all_dynamic_static_checks_passed'])
        report['status']=('completed_unstable' if not report['all_timings_stable'] else
            'completed_pending_review' if report['all_numerical_checks_passed'] else 'completed_with_numerical_differences')
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:atomic_json(args.output,report)
    print(report['status'])


if __name__=='__main__':main()
