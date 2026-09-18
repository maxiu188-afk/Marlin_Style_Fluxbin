import unittest
import torch
from fluxbin_style.distillation import HybridScaleLinear, distillation_loss, SCALES
from fluxbin_style import quantize_hybrid_two_base_obq, TwoBaseRankOneOptimizationConfig
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_qwen3_8b_hybrid_compensation_probe import payload_of
sys.path.pop(0)

class DistillationTests(unittest.TestCase):
    def test_packed_forward_gradients_and_scale_only_update(self):
        torch.manual_seed(71)
        w=torch.randn(6,16); cfg=TwoBaseRankOneOptimizationConfig(max_iters=2)
        payload=payload_of(quantize_hybrid_two_base_obq(w,torch.eye(16),group_size=8,columns_per_group=2,global_config=cfg,refinement_config=cfg))
        model=HybridScaleLinear(payload,group_size=8,columns_per_group=2)
        oracle=HybridScaleLinear(payload,group_size=8,columns_per_group=2)
        x=torch.randn(2,3,16,requires_grad=True); y=x.detach().clone().requires_grad_()
        actual=model(x);expected=torch.nn.functional.linear(y,oracle.reconstruct_weight(y.dtype))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.square().mean().backward();expected.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad)
        self.assertEqual(set(dict(model.named_parameters())),set(SCALES))
        for name,p in model.named_parameters():
            torch.testing.assert_close(p.grad,dict(oracle.named_parameters())[name].grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        before=model.export_payload()
        torch.optim.Adam(model.parameters(),lr=1e-6).step()
        after=model.export_payload()
        for k in before:
            if k not in SCALES:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(before[k],after[k]) for k in SCALES))
        restored=HybridScaleLinear(after,group_size=8,columns_per_group=2)
        with torch.no_grad():torch.testing.assert_close(model(x.bfloat16()),restored(x.bfloat16()),rtol=0,atol=0)

    def test_block_hooks_and_frozen_teacher(self):
        import copy
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from fluxbin_style.distillation import forward_losses
        teacher=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4)).eval()
        student=copy.deepcopy(teacher)
        teacher.requires_grad_(False)
        loss=forward_losses(teacher,student,torch.tensor([[1,2,3]]),{'ce':1.,'feature':1.},expected_layers=2)
        loss['total'].backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(not b._forward_hooks for m in (teacher,student) for b in m.model.layers))
        with self.assertRaises(ValueError):forward_losses(teacher,student,torch.tensor([[1,2,3]]),{'ce':0.,'feature':1.},expected_layers=2)
        self.assertTrue(all(not b._forward_hooks for m in (teacher,student) for b in m.model.layers))

    def test_two_losses_shift_layer_average_detach_and_normalization(self):
        ids=torch.tensor([[0,1,2]])
        logits=torch.randn(1,3,4,requires_grad=True)
        teachers=[torch.zeros(1,3,2,requires_grad=True) for _ in range(2)]
        students=[torch.ones(1,3,2,requires_grad=True),torch.full((1,3,2),3.,requires_grad=True)]
        loss=distillation_loss(logits,ids,teachers,students,{'ce':2.,'feature':5.},expected_layers=2)
        self.assertEqual(float(loss['feature'].detach()),5.)
        ce=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,4),ids[:,1:].reshape(-1))
        torch.testing.assert_close(loss['total'],ce/2+1)
        loss['total'].backward()
        self.assertTrue(all(t.grad is None for t in teachers))
        self.assertTrue(all(s.grad is not None for s in students))
        self.assertEqual(float(logits.grad[:,-1].abs().sum()),0.)
        with self.assertRaises(ValueError):distillation_loss(logits,ids,teachers,students,{'ce':0.,'feature':1.},expected_layers=2)
        with self.assertRaises(ValueError):distillation_loss(logits,ids,teachers,students,{'ce':1.,'feature':1.})
