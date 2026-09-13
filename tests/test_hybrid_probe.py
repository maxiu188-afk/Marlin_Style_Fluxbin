import json
import copy
import sys
import unittest
from pathlib import Path
import torch
from fluxbin_style import quantize_hybrid_two_base_obq,TwoBaseRankOneOptimizationConfig,materialize_hybrid_s8_weight
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_qwen3_8b_hybrid_compensation_probe as probe
sys.path.pop(0)


class ProbeTests(unittest.TestCase):
    def test_frozen_runtime_variants(self):
        base=json.loads(probe.CONFIG.read_text());pcie=json.loads(probe.PCIE_CONFIG.read_text())
        probe.validate_probe_config(base);probe.validate_probe_config(pcie)
        self.assertEqual({k:v for k,v in base.items() if k!="execution"},{k:v for k,v in pcie.items() if k!="execution"})
        for key,value in [("group_size",64),("damp_percent",.02)]:
            wrong=copy.deepcopy(pcie);wrong[key]=value
            with self.assertRaises(ValueError):probe.validate_probe_config(wrong)

    def test_original_prefix_replay_and_early_stop(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.first=torch.nn.Linear(4,4);self.last=torch.nn.Linear(4,4);self.completed=0
            def forward(self,input_ids,use_cache):
                x=torch.nn.functional.one_hot(input_ids,4).float();x=self.last(self.first(x).relu())
                self.completed+=1
                return x
        model=Tiny().eval(); tokens=torch.tensor([[0,1,2],[3,2,1]])
        groups={'first':['first'],'last':['last']};seen=[]
        a=probe.walk_prefix(model,tokens,groups,lambda n,x:seen.append((n,x.clone())),torch.device('cpu'))
        b=probe.walk_prefix(model,tokens,groups,lambda n,x:None,torch.device('cpu'))
        self.assertEqual(a,b);self.assertEqual(a['rows'],{'first':6,'last':6});self.assertEqual(model.completed,0)
        self.assertEqual(len(seen),4)
        self.assertFalse(model.first._forward_pre_hooks);self.assertFalse(model.last._forward_pre_hooks)

    def test_probe_payload_has_same_bf16_weight_as_fit(self):
        torch.manual_seed(4);w=torch.randn(9,16);cfg=TwoBaseRankOneOptimizationConfig(max_iters=3)
        fit=quantize_hybrid_two_base_obq(w,torch.eye(16),group_size=8,columns_per_group=2,global_config=cfg,refinement_config=cfg)
        payload=probe.payload_of(fit)
        q=materialize_hybrid_s8_weight(**payload,group_size=8,columns_per_group=2,device='cpu',output_dtype=torch.bfloat16)
        torch.testing.assert_close(q,fit.decomposition.reconstruct().bfloat16(),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
