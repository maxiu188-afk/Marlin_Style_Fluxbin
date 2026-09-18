import sys,unittest,copy
from pathlib import Path
import torch
from fluxbin_style import quantize_hybrid_two_base_obq,TwoBaseRankOneOptimizationConfig,tensor_sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_qwen3_8b_hybrid_distillation as r
from run_qwen3_8b_hybrid_compensation_probe import payload_of
from safetensors.torch import save_file
sys.path.pop(0)
import tempfile
class RunnerTests(unittest.TestCase):
    def test_install_freezes_non_scale_parameters(self):
        model=torch.nn.Module();model.model=torch.nn.Module();layer=torch.nn.Module();layer.linear=torch.nn.Linear(128,4,bias=False)
        model.model.layers=torch.nn.ModuleList([layer]);model.extra=torch.nn.Parameter(torch.ones(2))
        original=layer.linear.weight.detach().clone()
        fit=quantize_hybrid_two_base_obq(original,torch.eye(128),group_size=128,columns_per_group=8,global_config=TwoBaseRankOneOptimizationConfig(max_iters=1),refinement_config=TwoBaseRankOneOptimizationConfig(max_iters=1))
        payload=payload_of(fit)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'payload.safetensors';save_file({'linear.'+k:v for k,v in payload.items()},p)
            records=[dict(layer_index=0,payload_path=p,metadata={'linears':[{'module':'linear','target_bf16_sha256':tensor_sha256(original)}]})]
            modules=r.install_student(model,records)
        trainable,fixed=r.invariants(model,modules,{'expected_trainable_tensor_count':4,'expected_trainable_parameter_count':sum(payload[k].numel() for k in r.SCALES)})
        self.assertFalse(model.extra.requires_grad)
        self.assertEqual(len(fixed),3)
        self.assertEqual(len(trainable),4)
        x=torch.randn(1,2,128)
        layer.linear(x).sum().backward()
        self.assertTrue(all(p.grad is not None for p in trainable.values()))
