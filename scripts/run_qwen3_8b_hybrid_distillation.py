#!/usr/bin/env python3
"""Gated smoke / bounded sample200 scale-only distillation. Never evaluates test."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from fluxbin_style import atomic_json,sha256_file,tensor_sha256
from fluxbin_style.distillation import HybridScaleLinear,forward_losses,SCALES,FIXED
from run_qwen3_full_hessian_obq_s8_ppl import validate_arm_ledger
from run_qwen3_two_base_rank1_s8_ppl import validate_runtime
from distillation_reference import generate_sequences,filter_candidates,load_validation_protocol,validation_metrics

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiments/qwen3_8b_conditioned_distillation_sample200_v1.json'
PARENT=ROOT/'configs/evaluation/qwen3_8b_wikitext2_conditioned_hybrid_v1.json'

def sources():
    paths=list((ROOT/'src/fluxbin_style').glob('*.py'))+[Path(__file__),ROOT/'scripts/distillation_reference.py',ROOT/'scripts/run_qwen3_full_hessian_obq_s8_ppl.py',ROOT/'scripts/run_qwen3_two_base_rank1_s8_ppl.py',CONFIG,PARENT]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths}

def install_student(student,records):
    student.requires_grad_(False)
    modules={}
    for entry in records:
        layer=entry['layer_index'];payload=load_file(entry['payload_path'])
        for rec in entry['metadata']['linears']:
            name=f'model.layers.{layer}.{rec["module"]}'
            old=student.get_submodule(name)
            tensors={k:payload[rec['module']+'.'+k] for k in SCALES+FIXED}
            new=HybridScaleLinear(tensors,bias=old.bias).to(old.weight.device)
            if tensor_sha256(old.weight)!=rec['target_bf16_sha256']:raise ValueError('student parent target drift')
            parent,leaf=name.rsplit('.',1);setattr(student.get_submodule(parent),leaf,new)
            modules[name]=new
    return modules

def invariants(model,modules,expected):
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    names={n+'.'+s for n in modules for s in SCALES}
    if set(trainable)!=names or len(names)!=expected['expected_trainable_tensor_count'] or sum(p.numel() for p in trainable.values())!=expected['expected_trainable_parameter_count']:
        raise ValueError('trainable parameter inventory drifted')
    return trainable,{n+'.'+k:tensor_sha256(getattr(m,k)) for n,m in modules.items() for k in FIXED}

@torch.no_grad()
def mean_losses(teacher,student,seqs,norms,batch,device):
    student.eval();totals={'ce':0.,'feature':0.}
    for i in range(0,len(seqs),batch):
        ids=torch.tensor(seqs[i:i+batch],device=device)
        values=forward_losses(teacher,student,ids,norms)
        for k in totals:totals[k]+=float(values[k])*len(ids)
    values={k:v/len(seqs) for k,v in totals.items()}
    values['total']=values['ce']/norms['ce']+values['feature']/norms['feature']
    return values

def train_step(teacher,student,ids,norms,optimizer,clip):
    student.train();optimizer.zero_grad(set_to_none=True)
    loss=forward_losses(teacher,student,ids,norms);loss['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],clip,error_if_nonfinite=True)
    optimizer.step()
    return {**{k:float(v.detach()) for k,v in loss.items()},'gradient_norm':float(norm)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','train'],required=True)
    for name in ['snapshot-root','parent-result','parent-source','parent-artifact','parent-acceptance','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--validation-dataset',type=Path)
    p.add_argument('--smoke-result',type=Path)
    args=p.parse_args();c=json.loads(CONFIG.read_text());parent=json.loads(PARENT.read_text())
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    device=validate_runtime(c);source=sources()
    if sha256_file(args.parent_acceptance)!=parent['accepted_full_review_sha256']:raise ValueError('parent acceptance drift')
    if sha256_file(args.parent_result)!=c['student']['accepted_reconstruction_result_sha256']:raise ValueError('parent result drift')
    if args.snapshot_root.name!=c['teacher']['revision']:raise ValueError('snapshot revision drift')
    for name,h in parent['model_preflight_files'].items():
        if sha256_file(args.snapshot_root/name)!=h:raise ValueError('snapshot drift')
    _,records=validate_arm_ledger(parent,arm='hybrid_s8',result_path=args.parent_result,source_manifest_path=args.parent_source,artifact_dir=args.parent_artifact)
    if args.mode=='train':
        if args.smoke_result is None or args.validation_dataset is None:raise ValueError('accepted smoke and validation data required')
        smoke=json.loads(args.smoke_result.read_text())
        if smoke['status']!='passed' or smoke['source_files_sha256']!=source:raise ValueError('smoke is not passed for current source')
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir/'provenance.json',{'config':c,'source_files_sha256':source,'parent_result_sha256':sha256_file(args.parent_result),'mode':args.mode})
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    from transformers import AutoModelForCausalLM
    kwargs=dict(local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    teacher=AutoModelForCausalLM.from_pretrained(args.snapshot_root,**kwargs).to(device).eval().requires_grad_(False)
    student=AutoModelForCausalLM.from_pretrained(args.snapshot_root,**kwargs).to(device).eval()
    modules=install_student(student,records)
    parameters,fixed=invariants(student,modules,c['student'])
    scales_before={name:tensor_sha256(p) for name,p in parameters.items()}
    settings=c['distillation'];optimizer=torch.optim.Adam(list(parameters.values()),lr=settings['learning_rate'],betas=tuple(settings['betas']),eps=settings['epsilon'],weight_decay=0.)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    if args.mode=='smoke':
        ids=torch.randint(teacher.config.vocab_size,(1,128),device=device)
        with torch.no_grad():
            initial=forward_losses(teacher,student,ids,{'ce':1.,'feature':1.})
            norms={k:float(initial[k]) for k in ('ce','feature')}
            # Parent decoder and step0 student must materialize identical BF16 weights.
            for entry in records:
                payload=load_file(entry['payload_path'])
                for rec in entry['metadata']['linears']:
                    name=f'model.layers.{entry["layer_index"]}.{rec["module"]}'
                    from fluxbin_style import materialize_hybrid_s8_weight
                    expected=materialize_hybrid_s8_weight(**{k:payload[rec['module']+'.'+k] for k in SCALES+FIXED},group_size=128,columns_per_group=8,device=device,output_dtype=torch.bfloat16)
                    if not torch.equal(expected,modules[name].reconstruct_weight(torch.bfloat16)):raise ValueError('step0 decode mismatch')
        metrics=train_step(teacher,student,ids,norms,optimizer,settings['gradient_clip_norm'])
        _,after=invariants(student,modules,c['student'])
        if fixed!=after or any(p.grad is not None for p in teacher.parameters()):raise ValueError('frozen state changed')
        changes={k:sum(tensor_sha256(p)!=scales_before[n] for n,p in parameters.items() if n.endswith('.'+k)) for k in SCALES}
        if not all(changes.values()):raise ValueError('some scale families did not update')
        with torch.no_grad():
            forward_losses(teacher,student,ids.repeat(4,1),norms)
            teacher(input_ids=ids.repeat(32,1),use_cache=False,logits_to_keep=1)
            # Exercise the longest real-validation shape without accessing test data.
            validation_metrics(teacher,student,[ids.repeat(1,16)[0].cpu().tolist()],device=device)
        if sources()!=source:raise ValueError('source changed during smoke')
        result={'status':'passed','mode':'smoke','source_files_sha256':source,'metrics':metrics,'changed_scale_tensors':changes,'fixed_hashes_unchanged':True,'teacher_gradients_absent':True,'step0_decode_exact':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,'training_auto_launch':False}
        atomic_json(args.output_dir/'result.json',result);print(json.dumps(result),flush=True);return
    synth=c['synthetic_data']
    generated=[]
    for label,count,seed,batch in [('train',synth['generated_sample_count'],c['seed'],synth['generation_batch_size']),('validation',synth['validation_sample_count'],synth['validation_seed'],synth['validation_generation_batch_size'])]:
        generated.append(generate_sequences(teacher,sample_count=count,sequence_length=synth['sequence_length'],vocabulary_size=teacher.config.vocab_size,temperature=synth['temperature'],batch_size=batch,generator=torch.Generator(device=device).manual_seed(seed),device=device,label=label))
    candidates,validation=generated
    if len(set(map(tuple,candidates)))!=len(candidates) or len(set(map(tuple,validation)))!=len(validation) or set(map(tuple,candidates))&set(map(tuple,validation)):raise ValueError('duplicate or overlapping synthetic sequences')
    scores,selected=filter_candidates(teacher,student,candidates,keep_count=synth['filter_keep_count'],batch_size=1,device=device)
    train=[candidates[i] for i in selected]
    save_file({'candidates':torch.tensor(candidates,dtype=torch.int32),'validation':torch.tensor(validation,dtype=torch.int32),'selected_indices':torch.tensor(selected),'scores':torch.tensor(scores)},args.output_dir/'synthetic.safetensors')
    blocks,protocol=load_validation_protocol(c,args.validation_dataset,args.snapshot_root)
    atomic_json(args.output_dir/'validation_protocol.json',protocol)
    initial=mean_losses(teacher,student,train,{'ce':1.,'feature':1.},4,device);norms={k:initial[k] for k in ('ce','feature')}
    atomic_json(args.output_dir/'normalizers.json',norms)
    curves=[];real=[];steps=[]
    def monitor(step):
        if step%settings['monitor_interval_steps']==0:
            curves.append({'step':step,'train':mean_losses(teacher,student,train,norms,4,device),'validation':mean_losses(teacher,student,validation,norms,4,device)})
        if step in c['real_validation']['monitor_steps']:
            v=validation_metrics(teacher,student,blocks,device=device)
            if not v['metrics_valid']:raise ValueError('nonfinite validation')
            if step==0 and v['teacher_perplexity']>=v['student_perplexity']:raise ValueError('teacher not better at step0')
            real.append({'step':step,'metrics':v})
        atomic_json(args.output_dir/'progress.json',{'steps':steps,'synthetic_curve':curves,'real_curve':real})
    monitor(0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings['total_steps'],eta_min=0.)
    generator=torch.Generator().manual_seed(c['seed'])
    for epoch in range(settings['epochs']):
        for idx in torch.randperm(len(train),generator=generator).tolist():
            step=len(steps)+1
            metrics=train_step(teacher,student,torch.tensor([train[idx]],device=device),norms,optimizer,settings['gradient_clip_norm'])
            scheduler.step();steps.append({'step':step,'epoch':epoch,'selected_position':idx,**metrics,'lr':scheduler.get_last_lr()[0]})
            print(f'DISTILL_STEP={step}/{settings["total_steps"]} loss={metrics["total"]}',flush=True);monitor(step)
    if len(steps)!=400 or invariants(student,modules,c['student'])[1]!=fixed:raise ValueError('step count or fixed tensors drifted')
    if sources()!=source:raise ValueError('source changed during training')
    changes={k:sum(tensor_sha256(p)!=scales_before[n] for n,p in parameters.items() if n.endswith('.'+k)) for k in SCALES}
    if not all(changes.values()) or any(p.grad is not None for p in teacher.parameters()):raise ValueError('trainable/frozen gradient gate failed')
    final=args.output_dir/'payloads';final.mkdir();exports=[]
    for entry in records:
        layer=entry['layer_index'];payload={}
        for rec in entry['metadata']['linears']:
            name=f'model.layers.{layer}.{rec["module"]}'
            payload.update({rec['module']+'.'+k:v for k,v in modules[name].export_payload().items()})
        path=final/f'layer-{layer:03d}.safetensors';save_file(payload,path)
        loaded=load_file(path)
        if any(not torch.equal(v,loaded[k]) for k,v in payload.items()):raise ValueError('export roundtrip failed')
        exports.append({'layer':layer,'sha256':sha256_file(path)})
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'order_rng':generator.get_state(),'normalizers':norms,'step':len(steps)},args.output_dir/'final_training_state.pt')
    atomic_json(args.output_dir/'result.json',{'status':'completed_pending_review','source_files_sha256':source,'normalizers':norms,'steps':steps,'synthetic_curve':curves,'real_curve':real,'payloads':exports,'fixed_hashes_unchanged':True,'changed_scale_tensors':changes,'synthetic_loss_reduced':curves[-1]['train']['total']<curves[0]['train']['total'],'test_evaluation_count':0,'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':main()
