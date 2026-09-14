"""Synthetic CPU protocol tests; no real-model/GPU performance claims."""
import copy
import json
from pathlib import Path
import unittest
import torch
from fluxbin_style.full_model_trial import decode_trace, compare_trace, validate_routes
from fluxbin_style.trial_gates import validate_block_gate
from fluxbin_style.deployment_artifacts import MANIFEST_SHA


class FullModelTrialTests(unittest.TestCase):
    def test_cached_context_matches_full_prefix_and_replay(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(17)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
                       num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        cfg._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(cfg).eval()
        ids=torch.tensor([[1,3,7]])
        first=decode_trace(model,ids,steps=3)
        replay=decode_trace(model,ids,steps=3,forced_tokens=first['fed_tokens'])
        self.assertEqual(first['cache_length'],6)
        self.assertTrue(compare_trace(replay,first,logprob_tolerance=.05)['passed'])
        with torch.inference_mode():
            for i in range(4):
                prefix=torch.cat((ids,first['fed_tokens'][:,:i]),dim=1)
                expected=model(input_ids=prefix,use_cache=False).logits[:,-1,:]
                torch.testing.assert_close(first['logits'][:,i,:],expected,atol=1e-6,rtol=1e-5)
        wrong=copy.deepcopy(replay);wrong['fed_tokens'][0,0]+=1
        self.assertFalse(compare_trace(wrong,first,logprob_tolerance=.05)['passed'])
        wrong=copy.deepcopy(replay);wrong['logits'][0,0,0]=float('nan')
        self.assertFalse(compare_trace(wrong,first,logprob_tolerance=.05)['passed'])
        with self.assertRaises(ValueError):decode_trace(model,ids,steps=3,forced_tokens=torch.zeros(1,2))

    def test_routes_require_all_linears_and_no_decode_fallback(self):
        trace={'routes':{str(i):{'dense_fallback':1,'packed_m1':32} for i in range(252)}}
        self.assertTrue(validate_routes(trace,expected_linears=252,steps=32))
        trace['routes']['0']['dense_fallback']=2
        self.assertFalse(validate_routes(trace,expected_linears=252,steps=32))
        del trace['routes']['0']
        self.assertFalse(validate_routes(trace,expected_linears=252,steps=32))

    def test_block_gate_rejects_stale_incomplete_or_unstable_evidence(self):
        block={'status':'completed_pending_review','stage':'single_block_empty_cache_m1',
               'layer':0,'manifest_sha256':MANIFEST_SHA,'kernel_coverage':7,
               'source_sha256':{'a':'b'},'environment_sha256':'env','kernel':'v2_r2','groups_per_split':8,
               'checks':[{'passed':True,'repeat_exact':True} for _ in range(3)],
               'timings':{k:{'stable':True,'microseconds_per_call':[10.0]*7}
                          for k in ('original_bf16','decoded_step400_bf16','packed_step400')}}
        def gate(value):return validate_block_gate(value,source_sha256={'a':'b'},environment_sha256='env')
        self.assertEqual(gate(block),('v2_r2',8))
        for key,value in [('source_sha256',{}),('environment_sha256','old'),('checks',[]),('kernel_coverage',6)]:
            with self.assertRaises(ValueError):gate({**block,key:value})
        bad=copy.deepcopy(block);bad['timings']['packed_step400']['microseconds_per_call'][0]=100
        with self.assertRaises(ValueError):gate(bad)

    def test_36_layer_replacement_and_late_failure_before_mutation(self):
        from unittest.mock import patch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from fluxbin_style.deployment import PackedHybridLinear
        from fluxbin_style.deployment_artifacts import replace_model_linears
        from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
        from test_deployment import make_payload
        cfg=Qwen3Config(vocab_size=32,hidden_size=128,intermediate_size=128,
                       num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=32)
        model=Qwen3ForCausalLM(cfg).eval()
        payload={}
        for name in QWEN3_LINEAR_MODULES:
            linear=model.model.layers[0].get_submodule(name)
            payload.update({name+'.'+k:v for k,v in make_payload(linear.out_features,1).items()})
        # Synthetic tiny dimensions and payload IO only are mocked; real replacement is exercised.
        with patch('fluxbin_style.qwen3_8b.validate_architecture'), patch(
                'fluxbin_style.deployment_artifacts.load_accepted_layer',return_value=(payload,{'synthetic':True})):
            original=model.model.layers[-1].mlp.down_proj
            model.model.layers[-1].mlp.down_proj=torch.nn.Linear(128,127,bias=False)
            with self.assertRaisesRegex(ValueError,'layer 35'):
                replace_model_linears(model,Path('.'),kernel='v2_r2')
            self.assertFalse(any(isinstance(m,PackedHybridLinear) for m in model.modules()))
            model.model.layers[-1].mlp.down_proj=original
            coverage=replace_model_linears(model,Path('.'),kernel='v2_r2',groups_per_split=4,
                                           allow_prefill_fallback=True)
        packed=[m for m in model.modules() if isinstance(m,PackedHybridLinear)]
        self.assertEqual(len(packed),252)
        self.assertEqual(len(set(coverage['coverage'])),252)
        self.assertTrue(all(m.kernel=='v2_r2' and m.groups_per_split==4 and m.fallback=='dense' for m in packed))
        # CPU prefill is allowed; strict decode must fail, and finally must restore all policies.
        with self.assertRaisesRegex(ValueError,'unsupported input'):
            decode_trace(model,torch.tensor([[1,2,3]]),steps=2)
        self.assertTrue(all(m.fallback=='dense' for m in packed))
        self.assertTrue(all(m.route_counts['dense_fallback']==1 for m in packed))

    def test_report_only_records_difference_but_keeps_execution_guards(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/run_qwen3_8b_full_m1_trial.py'
        spec=importlib.util.spec_from_file_location('full_runner',path)
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        row={'coverage_passed':True,'correctness':{'passed':False,'fed_tokens_equal':True,'logits':{'passed':False}}}
        self.assertTrue(runner.must_abort(row,'strict'))
        self.assertFalse(runner.must_abort(row,'report-only'))
        row['coverage_passed']=False
        self.assertTrue(runner.must_abort(row,'report-only'))
        row['coverage_passed']=True;row['correctness']['fed_tokens_equal']=False
        self.assertTrue(runner.must_abort(row,'report-only'))
        row['correctness']['fed_tokens_equal']=True
        row['correctness']['logits']['reason']='nonfinite output'
        self.assertTrue(runner.must_abort(row,'report-only'))

    def test_candidate_suite_is_bounded_and_distinct(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/acceleration/m1_candidates_v1.json').read_text())
        self.assertEqual(len(cfg['candidates']),6)
        self.assertEqual(len({(c['kernel'],c['groups_per_split']) for c in cfg['candidates']}),6)
        self.assertEqual(cfg['modes'],['eager','graph'])
        self.assertLessEqual(cfg['timeout_seconds_per_trial'],300)


if __name__=='__main__':unittest.main()
