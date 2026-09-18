"""Prepared cache-ready Qwen3 decode; auditing is outside performance timing.

Static cache grows through an actual fixed continuation. A sequence Graph
contains every decode step, including embedding, attention, LM head and argmax.
Prefill and restoring the real prefix KV are excluded and reported separately.
"""
from contextlib import contextmanager
import time
import torch
from transformers import StaticCache
from .full_model_trial import packed_modules


@contextmanager
def prepared_linears(model, dtype):
    modules = packed_modules(model)
    if any(m._decode_binding is not None for m in modules.values()):
        raise ValueError('prepared decode contexts cannot overlap')
    try:
        for m in modules.values():m.bind_prepared_decode(dtype)
        yield modules
    finally:
        for m in modules.values():m.clear_prepared_decode()


class StaticDecodeSession:
    def __init__(self, model, input_ids, forced_tokens):
        if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.shape[1]<2:
            raise ValueError('batch1 nonempty real prefix required')
        if forced_tokens.ndim!=2 or forced_tokens.shape[0]!=1 or forced_tokens.shape[1]<1:
            raise ValueError('batch1 continuation required')
        if model.training:raise ValueError('eval model required')
        if any(m._decode_binding is not None for m in packed_modules(model).values()):
            raise ValueError('prepare real prefix before binding decode-only linears')
        if set(model.config.layer_types)!={'full_attention'}:
            raise ValueError('this protocol supports full-attention Qwen3 only')
        self.model = model
        self.ids = input_ids
        self.tokens = forced_tokens.to(input_ids.device).contiguous()
        self.steps = self.tokens.shape[1]
        self.prefix_length = input_ids.shape[1]
        self.maximum_length = self.prefix_length+self.steps
        self.cache = StaticCache(config=model.config,max_cache_len=self.maximum_length)
        with torch.inference_mode():
            out = model(input_ids=input_ids,past_key_values=self.cache,use_cache=True,logits_to_keep=1)
            self.prefill_logits = out.logits[:,-1,:].detach().clone()
            self.prefix = [(layer.keys[:,:,:self.prefix_length].clone(),
                            layer.values[:,:,:self.prefix_length].clone()) for layer in self.cache.layers]
        self.positions = torch.arange(self.prefix_length,self.maximum_length,
                                      device=input_ids.device).view(1,-1)
        # Boolean SDPA mask: True means attend. Static masks bypass dynamic mask
        # construction and never read a CUDA sequence length into Python.
        columns = torch.arange(self.maximum_length,device=input_ids.device)
        self.masks = [(columns<=p).view(1,1,1,-1) for p in range(self.prefix_length,self.maximum_length)]
        self.graph = None
        self.graph_outputs = None
        self.capture_routes = None

    @torch.inference_mode()
    def reset(self):
        for layer,(k,v) in zip(self.cache.layers,self.prefix):
            layer.keys.zero_();layer.values.zero_()
            layer.keys[:,:,:self.prefix_length].copy_(k)
            layer.values[:,:,:self.prefix_length].copy_(v)
            layer.cumulative_length.fill_(self.prefix_length)

    @torch.inference_mode()
    def run(self):
        outputs = []
        for i in range(self.steps):
            out = self.model(input_ids=self.tokens[:,i:i+1],past_key_values=self.cache,
                position_ids=self.positions[:,i:i+1],attention_mask={'full_attention':self.masks[i]},
                use_cache=True,logits_to_keep=1)
            logits = out.logits[:,-1,:]
            outputs.append((logits,logits.argmax(-1,keepdim=True)))
        return outputs

    def set_audit(self, enabled):
        for m in packed_modules(self.model).values():
            m._decode_audit = enabled
            m.route_counts = {'dense_fallback':0,'packed_m1':0}

    def routes(self):
        return {n:dict(m.route_counts) for n,m in packed_modules(self.model).items()}

    def trace(self, outputs):
        length = int(self.cache.get_seq_length())
        if length!=self.maximum_length:raise RuntimeError('static KV length mismatch')
        logits = torch.stack([self.prefill_logits]+[p[0] for p in outputs],dim=1).detach().cpu()
        predictions = torch.cat([self.prefill_logits.argmax(-1,keepdim=True)]+[p[1] for p in outputs],dim=1).cpu()
        if not torch.isfinite(logits).all():raise RuntimeError('nonfinite static decode logits')
        return dict(logits=logits,predictions=predictions,fed_tokens=self.tokens.cpu(),
                    cache_length=length,routes=self.routes())

    @torch.inference_mode()
    def audit(self):
        self.reset();self.set_audit(True)
        try:return self.trace(self.run())
        finally:self.set_audit(False)

    @torch.inference_mode()
    def capture(self):
        if self.ids.device.type!='cuda':raise RuntimeError('CUDA Graph requires NVIDIA CUDA')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):self.reset();self.run()
        torch.cuda.current_stream().wait_stream(stream)
        self.reset();torch.cuda.synchronize()
        self.set_audit(True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph,stream=stream):outputs=self.run()
            self.capture_routes = self.routes()
        finally:self.set_audit(False)
        self.graph,self.graph_outputs = graph,outputs
        self.reset();graph.replay();torch.cuda.synchronize()
        return self.trace(outputs)

    @torch.inference_mode()
    def measure(self, mode):
        if mode not in ('prepared_eager','sequence_graph'):raise ValueError('unknown mode')
        if mode=='sequence_graph' and self.graph is None:raise ValueError('capture first')
        self.reset()
        cuda = self.ids.device.type=='cuda'
        if cuda:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        wall=time.perf_counter()
        if cuda:start.record()
        if mode=='sequence_graph':
            self.graph.replay();outputs=self.graph_outputs
        else:outputs=self.run()
        if cuda:end.record();end.synchronize()
        wall_ms=(time.perf_counter()-wall)*1000
        # Checks are after the timer; keep outputs live equally across all arms.
        trace=self.trace(outputs)
        return dict(wall_ms=wall_ms,device_ms=start.elapsed_time(end) if cuda else None),trace
