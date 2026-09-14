import unittest
import torch
from torch import nn
from fluxbin_style.deployment import (
    FIELDS, PackedHybridLinear, convert_artifact, restore_artifact,
    decode_layout, m1_out, workspace_shape, load_extension, replace_block_linears,
)
from fluxbin_style.evaluation import materialize_hybrid_s8_weight
from fluxbin_style.packing import pack_two_bases


def make_payload(o=7, g=3, seed=19):
    rng = torch.Generator().manual_seed(seed)
    def signs(k):
        return pack_two_bases(torch.randint(0, 2, (2, o, k), generator=rng).mul(2).sub(1))
    return dict(zip(FIELDS, (
        signs(g*128), torch.randn(2,o,g,generator=rng)*.1,
        torch.randn(2,g,128,generator=rng),
        torch.stack([torch.randperm(128,generator=rng)[:8].sort().values for _ in range(g)]).short(),
        signs(g*8), torch.randn(2,o,g,generator=rng)*.07,
        torch.randn(2,g,8,generator=rng),
    )))


def oracle(p, dtype):
    return materialize_hybrid_s8_weight(**p, group_size=128, columns_per_group=8,
                                       device=p['global_sign_codes'].device, output_dtype=dtype)


class DeploymentTests(unittest.TestCase):
    def test_lossless_conversion_and_combined_weight_rounding(self):
        for o,g in ((1,1),(7,3),(17,10)):
            p = make_payload(o,g)
            q = convert_artifact(p)
            back = restore_artifact(q)
            for key in FIELDS:
                self.assertTrue(torch.equal(p[key],back[key]), key)
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                self.assertTrue(torch.equal(decode_layout(q,dtype),oracle(p,dtype)))
            expected = torch.full((g,128),-1,dtype=torch.int16)
            expected.scatter_(1,p['refinement_indices'].long(),torch.arange(8,dtype=torch.int16).expand(g,8))
            self.assertTrue(torch.equal(expected,q['lookup']))
            # Conversion is independent of source mutation.
            saved = q['rows'].clone()
            p['global_row_scales'].zero_()
            self.assertTrue(torch.equal(saved,q['rows']))

    def test_fail_closed_payload_contract(self):
        for field, mutate in (
            ('refinement_indices', lambda x: x.fill_(128)),
            ('refinement_indices', lambda x: x.zero_()),
            ('global_row_scales', lambda x: x.fill_(float('nan'))),
            ('global_column_scales', lambda x: x.half()),
            ('refinement_sign_codes', lambda x: x.long()),
        ):
            p=make_payload();p[field]=mutate(p[field])
            with self.assertRaises(ValueError):convert_artifact(p)
        with self.assertRaises(ValueError):convert_artifact({})

    def test_explicit_fallback_and_shape_preservation(self):
        p=make_payload();x=torch.randn(1,2,384)
        strict=PackedHybridLinear(p)
        with torch.inference_mode():
            with self.assertRaises(ValueError):strict(x)
            layer=PackedHybridLinear(p,fallback='dense')
            torch.testing.assert_close(layer(x),torch.nn.functional.linear(x,oracle(p,x.dtype)))
            self.assertEqual(layer.last_route,'dense_fallback')
        with self.assertRaises(RuntimeError):layer(x)
        layer.half()
        with torch.inference_mode(), self.assertRaises(ValueError):layer(x.half())

    def test_block_adapter_checks_before_mutation(self):
        from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
        block=nn.Module();block.self_attn=nn.Module();block.mlp=nn.Module();payload={}
        for name in QWEN3_LINEAR_MODULES:
            parent,attr=name.split('.')
            setattr(block.get_submodule(parent),attr,nn.Linear(128,7,bias=False))
            payload.update({name+'.'+k:v for k,v in make_payload(7,1).items()})
        broken=dict(payload);broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError):replace_block_linears(block,broken)
        self.assertTrue(all(isinstance(block.get_submodule(n),nn.Linear) for n in QWEN3_LINEAR_MODULES))
        self.assertEqual(len(replace_block_linears(block,payload,fallback='dense')),7)
        with torch.inference_mode():
            self.assertEqual(tuple(block.self_attn.q_proj(torch.zeros(1,128)).shape),(1,7))


    def test_real_qwen3_block_api_with_tiny_synthetic_weights(self):
        import copy
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
        from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
        config=Qwen3Config(hidden_size=128,intermediate_size=256,num_hidden_layers=1,
                           num_attention_heads=4,num_key_value_heads=2,head_dim=32)
        config._attn_implementation='sdpa'
        block=Qwen3DecoderLayer(config,0).to(torch.bfloat16).eval()
        payload={}
        with torch.no_grad():
            for name in QWEN3_LINEAR_MODULES:
                linear=block.get_submodule(name)
                p=make_payload(linear.out_features,linear.in_features//128)
                linear.weight.copy_(oracle(p,torch.bfloat16))
                payload.update({name+'.'+k:v for k,v in p.items()})
        candidate=copy.deepcopy(block)
        replace_block_linears(candidate,payload,fallback='dense')
        rotary=Qwen3RotaryEmbedding(config)
        x=torch.randn(1,1,128,dtype=torch.bfloat16)
        position=torch.zeros(1,1,dtype=torch.long)
        with torch.inference_mode():
            kwargs={'position_ids':position,'position_embeddings':rotary(x,position),'use_cache':False}
            self.assertTrue(torch.equal(block(x,**kwargs),candidate(x,**kwargs)))
        self.assertTrue(all(m.rows.dtype==torch.float32 for m in candidate.modules()
                            if isinstance(m,PackedHybridLinear)))


@unittest.skipUnless(torch.cuda.is_available(), 'requires NVIDIA CUDA compiler/device')
class CUDADeploymentTests(unittest.TestCase):
    def test_oracle_tail_split_stream_graph_and_repeated_calls(self):
        load_extension()
        for o,g in ((1,1),(7,3),(17,10)):
            for dtype in (torch.bfloat16,torch.float16):
                p={k:v.cuda() for k,v in make_payload(o,g).items()}
                q=convert_artifact(p);x=torch.randn(1,g*128,device='cuda',dtype=dtype)
                ref=torch.nn.functional.linear(x,oracle(p,dtype))
                for gps in (1,8):
                    w=torch.empty(workspace_shape(o,g*128,gps),device='cuda')
                    y=torch.empty_like(ref)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):m1_out(x,q,y,w,groups_per_split=gps)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.testing.assert_close(y,ref,atol=.02,rtol=.02)
                    saved=y.clone();w.fill_(float('nan'))
                    m1_out(x,q,y,w,groups_per_split=gps)
                    self.assertTrue(torch.equal(y,saved))
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):m1_out(x,q,y,w,groups_per_split=gps)
                    graph.replay();torch.cuda.synchronize()
                    self.assertTrue(torch.equal(y,saved))
                    with self.assertRaises(RuntimeError):m1_out(x.expand(2,-1).contiguous(),q,y,w)


if __name__=='__main__':unittest.main()
