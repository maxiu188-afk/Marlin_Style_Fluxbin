"""Small HF cached-decode protocol shared by the full-model runner and CPU tests."""
from __future__ import annotations
import time
import torch
from .deployment import PackedHybridLinear
from .acceleration_checks import numerical_gate


def packed_modules(model):
    return {name:m for name,m in model.named_modules() if isinstance(m,PackedHybridLinear)}


@torch.inference_mode()
def decode_trace(model, input_ids, *, steps, forced_tokens=None):
    """Prefill once, then exactly `steps` M=1 calls on one growing dynamic KV cache.

    Returns steps+1 next-token predictions, including the prefill prediction.
    forced_tokens supplies the `steps` fed tokens, enabling identical contexts.
    CPU is supported only for synthetic tests; the real runner requires CUDA.
    """
    if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.shape[1]<2 or steps<1:
        raise ValueError('batch1, prompt length>=2 and positive decode steps required')
    if forced_tokens is not None and tuple(forced_tokens.shape)!=(1,steps):
        raise ValueError('forced continuation shape must be [1,steps]')
    forced=None if forced_tokens is None else forced_tokens.to(input_ids.device)
    modules=packed_modules(model)
    previous={n:m.fallback for n,m in modules.items()}
    for m in modules.values():
        m.route_counts={'packed_m1':0,'dense_fallback':0};m.fallback='dense'
    cuda=input_ids.is_cuda
    def sync():
        if cuda:torch.cuda.synchronize(input_ids.device)
    def event():return torch.cuda.Event(enable_timing=True) if cuda else None
    sync();start_wall=time.perf_counter();pre_start,pre_end=event(),event()
    logits=[];predictions=[];fed=[];step_events=[]
    try:
        if cuda:pre_start.record()
        output=model(input_ids=input_ids,use_cache=True,logits_to_keep=1)
        cache=output.past_key_values
        logits.append(output.logits[:,-1,:].detach())
        current=logits[-1].argmax(-1,keepdim=True);predictions.append(current)
        if cuda:pre_end.record()
        sync();prefill_wall=(time.perf_counter()-start_wall)*1000
        for m in modules.values():m.fallback='error'
        decode_start,decode_end=event(),event()
        start_wall=time.perf_counter()
        if cuda:decode_start.record()
        for i in range(steps):
            token=current if forced is None else forced[:,i:i+1]
            fed.append(token)
            begin,end=event(),event()
            if cuda:begin.record()
            output=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
            logits.append(output.logits[:,-1,:].detach())
            current=logits[-1].argmax(-1,keepdim=True);predictions.append(current)
            if cuda:end.record();step_events.append((begin,end))
        if cuda:decode_end.record()
        sync();decode_wall=(time.perf_counter()-start_wall)*1000
        length=cache.get_seq_length()
        length=int(length.item()) if isinstance(length,torch.Tensor) else int(length)
        if length!=input_ids.shape[1]+steps:raise RuntimeError('KV cache length drift')
        return {'logits':torch.stack(logits,dim=1).cpu(),
                'predictions':torch.cat(predictions,dim=1).cpu(),
                'fed_tokens':torch.cat(fed,dim=1).cpu(),'cache_length':length,
                'prefill_wall_ms':prefill_wall,'decode_wall_ms':decode_wall,
                'prefill_device_ms':pre_start.elapsed_time(pre_end) if cuda else None,
                'decode_device_ms':decode_start.elapsed_time(decode_end) if cuda else None,
                'step_device_ms':[a.elapsed_time(b) for a,b in step_events],
                'routes':{n:dict(m.route_counts) for n,m in modules.items()}}
    finally:
        for n,m in modules.items():m.fallback=previous[n]


def compare_trace(candidate, reference, *, logprob_tolerance):
    if logprob_tolerance<0:raise ValueError('negative logprob tolerance')
    gate=numerical_gate(candidate['logits'],reference['logits'])
    if candidate['logits'].shape!=reference['logits'].shape:return {'passed':False,'reason':'shape drift'}
    tokens_equal=torch.equal(candidate['predictions'],reference['predictions'])
    contexts_equal=torch.equal(candidate['fed_tokens'],reference['fed_tokens'])
    delta=(candidate['logits'].float().log_softmax(-1)-reference['logits'].float().log_softmax(-1)).abs()
    maximum=float(delta.max());finite=bool(torch.isfinite(delta).all())
    return {'passed':gate['passed'] and tokens_equal and contexts_equal and finite and maximum<=logprob_tolerance,
            'logits':gate,'greedy_tokens_equal':tokens_equal,'fed_tokens_equal':contexts_equal,
            'max_abs_logprob_error':maximum,'logprob_tolerance':logprob_tolerance}


def validate_routes(trace, *, expected_linears, steps):
    return len(trace['routes'])==expected_linears and all(
        r=={'dense_fallback':1,'packed_m1':steps} for r in trace['routes'].values())
