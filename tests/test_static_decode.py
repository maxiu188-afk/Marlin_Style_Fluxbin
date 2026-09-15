"""CPU context/contract tests; CUDA tests must run on the experiment GPU."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from fluxbin_style.full_model_trial import decode_trace
from fluxbin_style.static_decode import StaticDecodeSession, prepared_linears
from fluxbin_style.deployment import PackedHybridLinear
from test_deployment import make_payload


class StaticDecodeTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(17)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        cfg._attn_implementation='sdpa'
        return Qwen3ForCausalLM(cfg).eval()

    def test_static_matches_dynamic_and_full_prefix_and_reset(self):
        model=self.model();ids=torch.tensor([[1,3,7]])
        dynamic=decode_trace(model,ids,steps=3)
        session=StaticDecodeSession(model,ids,dynamic['fed_tokens'])
        pointers=[(l.keys.data_ptr(),l.values.data_ptr()) for l in session.cache.layers]
        first=session.audit()
        torch.testing.assert_close(first['logits'],dynamic['logits'],atol=1e-6,rtol=1e-5)
        for _ in range(2):
            timing,trace=session.measure('prepared_eager')
            self.assertGreater(timing['wall_ms'],0)
            self.assertIsNone(timing['device_ms'])
            self.assertTrue(torch.equal(trace['logits'],first['logits']))
            self.assertEqual(trace['cache_length'],6)
        self.assertEqual(pointers,[(l.keys.data_ptr(),l.values.data_ptr()) for l in session.cache.layers])
        with torch.inference_mode():
            session.reset()
            # Future cache slots must be masked even if they contain stale values.
            for layer in session.cache.layers:
                layer.keys[:,:,4:].fill_(100);layer.values[:,:,4:].fill_(100)
            poisoned=session.trace(session.run())
            torch.testing.assert_close(poisoned['logits'],first['logits'],atol=0,rtol=0)
            for i in range(4):
                prefix=torch.cat((ids,dynamic['fed_tokens'][:,:i]),1)
                expected=model(input_ids=prefix,use_cache=False).logits[:,-1,:]
                torch.testing.assert_close(first['logits'][:,i,:],expected,atol=1e-6,rtol=1e-5)
        with self.assertRaisesRegex(RuntimeError,'CUDA'):session.capture()
        with self.assertRaisesRegex(ValueError,'capture first'):session.measure('sequence_graph')

    def test_binding_rejects_cpu_and_cleanup_on_partial_failure(self):
        modules=torch.nn.Sequential(PackedHybridLinear(make_payload(32,1)),PackedHybridLinear(make_payload(32,1)))
        with self.assertRaisesRegex(ValueError,'CUDA'):modules[0].bind_prepared_decode(torch.bfloat16)
        def partial(module,dtype):
            module._decode_binding=lambda x:x
            if module is modules[1]:raise RuntimeError('partial failure')
        with patch.object(PackedHybridLinear,'bind_prepared_decode',partial):
            with self.assertRaisesRegex(RuntimeError,'partial failure'):
                with prepared_linears(modules,torch.bfloat16):pass
        self.assertTrue(all(m._decode_binding is None for m in modules))
        modules[0]._decode_binding=lambda x:x
        modules[0].to('cpu')
        self.assertIsNone(modules[0]._decode_binding)

    @unittest.skipUnless(torch.cuda.is_available(),'requires NVIDIA CUDA; no local performance claim')
    def test_cuda_prepared_binding_and_graph_replay(self):
        for kernel in ('v3','v4_late','v5_p1024'):
            with self.subTest(kernel=kernel),torch.inference_mode():
                module=PackedHybridLinear(make_payload(128,2),kernel=kernel,groups_per_split=1).cuda()
                model=torch.nn.Sequential(module)
                x=torch.randn(1,1,256,device='cuda',dtype=torch.bfloat16)
                expected=module(x).clone()
                with prepared_linears(model,torch.bfloat16):
                    module._decode_audit=True
                    module.route_counts={'packed_m1':0,'dense_fallback':0}
                    actual=module(x)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                    self.assertEqual(module.route_counts['packed_m1'],1)
                    pointer=actual.data_ptr();module._decode_audit=False
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):module(x)
                    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):output=module(x)
                    x.mul_(.5);graph.replay();torch.cuda.synchronize()
                    result=output.clone()
                    self.assertEqual(output.data_ptr(),pointer)
                    with self.assertRaises(ValueError):module(x.view(1,256))
                torch.testing.assert_close(result,module(x),atol=0,rtol=0)
                self.assertIsNone(module._decode_output)

    @unittest.skipUnless(torch.cuda.is_available(),'requires NVIDIA CUDA')
    def test_cuda_full_sequence_graph_matches_eager(self):
        model=self.model().cuda().to(torch.bfloat16)
        ids=torch.tensor([[1,3,7]],device='cuda')
        dynamic=decode_trace(model,ids,steps=3)
        session=StaticDecodeSession(model,ids,dynamic['fed_tokens'])
        expected=session.audit()
        graph=session.capture()
        self.assertTrue(torch.equal(expected['logits'],graph['logits']))
        for mode in ('prepared_eager','sequence_graph','sequence_graph'):
            _,trace=session.measure(mode)
            self.assertTrue(torch.equal(expected['logits'],trace['logits']))

    @unittest.skipUnless(torch.cuda.is_available(),'requires NVIDIA CUDA')
    def test_cuda_packed_model_static_graph_and_shared_output_lifetime(self):
        from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
        from fluxbin_style.full_model_trial import compare_trace
        cfg=Qwen3Config(vocab_size=64,hidden_size=128,intermediate_size=256,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=32)
        cfg._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
        for layer in model.model.layers:
            for name in QWEN3_LINEAR_MODULES:
                linear=layer.get_submodule(name)
                parent,child=name.rsplit('.',1)
                setattr(layer.get_submodule(parent),child,PackedHybridLinear(
                    make_payload(linear.out_features,linear.in_features//128),
                    kernel='v5_p1024',groups_per_split=1,fallback='dense').cuda())
        ids=torch.tensor([[1,3,7]],device='cuda')
        dynamic=decode_trace(model,ids,steps=3)
        session=StaticDecodeSession(model,ids,dynamic['fed_tokens'])
        with prepared_linears(model,torch.bfloat16):
            expected=session.audit()
            self.assertTrue(compare_trace(expected,dynamic,logprob_tolerance=.05)['passed'])
            self.assertEqual(len(expected['routes']),14)
            self.assertTrue(all(r=={'dense_fallback':0,'packed_m1':3} for r in expected['routes'].values()))
            graph=session.capture()
            self.assertTrue(torch.equal(expected['logits'],graph['logits']))
            for mode in ('prepared_eager','sequence_graph','sequence_graph'):
                _,trace=session.measure(mode)
                self.assertTrue(torch.equal(expected['logits'],trace['logits']))

    def test_protocol_preserves_inputs_and_separates_modes(self):
        root=Path(__file__).resolve().parents[1]
        old=json.loads((root/'configs/acceleration/qwen3_8b_full_m1_v1.json').read_text())
        new=json.loads((root/'configs/acceleration/qwen3_8b_full_m1_v2.json').read_text())
        for key in ('prompts','seed','decode_steps','arms','max_prompt_tokens','logprob_max_abs_tolerance'):
            self.assertEqual(new[key],old[key])
        self.assertEqual(new['warmup'],8);self.assertEqual(new['repeats'],10)
        self.assertEqual(new['max_relative_timing_range'],.05)


class PreparedRunnerGuards(unittest.TestCase):
    def test_timing_and_repeat_guards(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/run_qwen3_8b_prepared_m1_trial.py'
        spec=importlib.util.spec_from_file_location('prepared_runner',path)
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        self.assertTrue(runner.statistics_row([10.,10.1],.05)['stable'])
        self.assertFalse(runner.statistics_row([10.,12.],.05)['stable'])
        for samples in ([],[float('nan')],[0.],[-1.]):
            with self.assertRaises(ValueError):runner.statistics_row(samples,.05)
        routes={str(i):{'dense_fallback':0,'packed_m1':32} for i in range(252)}
        runner.require_routes(routes,32)
        routes['0']['dense_fallback']=1
        with self.assertRaises(RuntimeError):runner.require_routes(routes,32)
        trace=dict(cache_length=5,logits=torch.ones(1),predictions=torch.ones(1),fed_tokens=torch.ones(1))
        runner.exact_trace(trace,trace)
        with self.assertRaises(RuntimeError):runner.exact_trace({**trace,'cache_length':6},trace)
        with self.assertRaises(RuntimeError):runner.exact_trace({**trace,'logits':torch.zeros(1)},trace)


if __name__=='__main__':unittest.main()
