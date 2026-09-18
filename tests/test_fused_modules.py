import collections
import unittest

import torch
import torch._dynamo
import torch._inductor.metrics as inductor_metrics
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3Config
from transformers.models.qwen3 import modeling_qwen3

from fluxbin_style.fused_modules import (
    fused_apply_rotary_pos_emb,
    fused_qwen3_modules,
    rms_norm_reference,
)

# narrow/slice/view style ops are metadata only and issue no kernel.
VIEW_OPS = {
    "view", "_unsafe_view", "slice", "narrow", "unsqueeze", "squeeze",
    "transpose", "permute", "expand", "reshape", "detach", "t", "select", "alias",
}


class OpCounter(TorchDispatchMode):
    def __init__(self):
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))


def compute_ops(call):
    with OpCounter() as counter:
        call()
    return sum(count for op, count in counter.ops.items() if op not in VIEW_OPS)


def rope_inputs(dtype=torch.bfloat16):
    torch.manual_seed(20260914)
    return (
        torch.randn(1, 32, 1, 128, dtype=dtype),
        torch.randn(1, 8, 1, 128, dtype=dtype),
        torch.randn(1, 1, 128, dtype=dtype),
        torch.randn(1, 1, 128, dtype=dtype),
    )


class FusedRopeTest(unittest.TestCase):
    def test_halves_the_kernel_count(self):
        q, k, cos, sin = rope_inputs()
        stock = compute_ops(lambda: modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin))
        fused = compute_ops(lambda: fused_apply_rotary_pos_emb(q, k, cos, sin))
        self.assertEqual(stock, 10)
        self.assertEqual(fused, 5)

    def test_matches_stock_within_bf16_rounding(self):
        for dtype, limit in ((torch.float32, 0.0), (torch.bfloat16, 2 ** -7)):
            q, k, cos, sin = rope_inputs(dtype)
            expected = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
            actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
            for got, want in zip(actual, expected):
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, want.dtype)
                deviation = (got.float() - want.float()).abs().max().item()
                self.assertLessEqual(deviation, limit * want.float().abs().max().item())

    def test_rejects_mismatched_inputs(self):
        q, k, cos, sin = rope_inputs()
        with self.assertRaises(ValueError):
            fused_apply_rotary_pos_emb(q, k.to(torch.float32), cos, sin)
        with self.assertRaises(ValueError):
            fused_apply_rotary_pos_emb(q, k[..., :64], cos, sin)


class FusedRmsNormTest(unittest.TestCase):
    def test_reference_is_bit_exact_with_stock(self):
        # The compiled path fuses this function, so it must reproduce stock
        # arithmetic exactly before compilation is involved at all.
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(20260914)
            norm = modeling_qwen3.Qwen3RMSNorm(4096).to(dtype)
            hidden = torch.randn(1, 1, 4096, dtype=dtype)
            expected = modeling_qwen3.Qwen3RMSNorm.forward(norm, hidden)
            actual = rms_norm_reference(hidden, norm.weight, norm.variance_epsilon)
            self.assertTrue(torch.equal(actual, expected))

    def test_functional_rms_norm_would_not_have_fused(self):
        # Guards the reason this module compiles instead of calling F.rms_norm:
        # aten::rms_norm is CompositeImplicitAutograd and decomposes unchanged.
        torch.manual_seed(20260914)
        weight = torch.randn(4096)
        hidden = torch.randn(1, 1, 4096)
        self.assertEqual(
            compute_ops(lambda: torch.nn.functional.rms_norm(hidden, (4096,), weight, 1e-6)),
            compute_ops(lambda: rms_norm_reference(hidden, weight, 1e-6)),
        )

    def test_inductor_emits_fewer_kernels_than_eager_ops(self):
        torch.manual_seed(20260914)
        weight = torch.randn(4096)
        hidden = torch.randn(1, 1, 4096)
        eager_ops = compute_ops(lambda: rms_norm_reference(hidden, weight, 1e-6))
        torch._dynamo.reset()
        inductor_metrics.reset()
        compiled = torch.compile(rms_norm_reference, dynamic=False)
        result = compiled(hidden, weight, 1e-6)
        kernels = inductor_metrics.generated_kernel_count
        self.assertGreater(kernels, 0)
        self.assertLess(kernels, eager_ops)
        deviation = (result - rms_norm_reference(hidden, weight, 1e-6)).abs().max().item()
        self.assertLess(deviation, 1e-5)


class WholeModelOpCountTest(unittest.TestCase):
    """End-to-end check on a tiny Qwen3 that the decode step really loses ops.

    Counts a real decode step rather than a single module, so a fusion that only
    looks good in isolation cannot pass. The compiled RMSNorm is excluded here:
    a TorchDispatchMode forces torch.compile back to eager, so its reduction is
    measured separately in FusedRmsNormTest.
    """

    LAYERS = 4

    def build(self):
        config = Qwen3Config(
            hidden_size=128, intermediate_size=256, num_hidden_layers=self.LAYERS,
            num_attention_heads=8, num_key_value_heads=2, head_dim=16,
            vocab_size=256, max_position_embeddings=64, attn_implementation="sdpa",
        )
        torch.manual_seed(20260914)
        model = modeling_qwen3.Qwen3ForCausalLM(config).eval()
        with torch.inference_mode():
            cache = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values

        def decode():
            with torch.inference_mode():
                model(input_ids=torch.tensor([[5]]), past_key_values=cache, use_cache=True)

        return model, decode

    def test_rope_fusion_removes_five_ops_per_layer(self):
        model, decode = self.build()
        stock = compute_ops(decode)
        with fused_qwen3_modules(rms_norm=False):
            fused = compute_ops(decode)
        self.assertEqual(stock - fused, 5 * self.LAYERS)

    def test_stock_rms_norm_dominates_the_non_linear_op_count(self):
        # The premise of compiling RMSNorm: it is the majority of the tail.
        model, decode = self.build()
        calls = []
        original = modeling_qwen3.Qwen3RMSNorm.forward
        try:
            modeling_qwen3.Qwen3RMSNorm.forward = lambda self, x: (
                calls.append(1) or original(self, x)
            )
            total = compute_ops(decode)
        finally:
            modeling_qwen3.Qwen3RMSNorm.forward = original
        self.assertEqual(len(calls), 4 * self.LAYERS + 1)
        self.assertGreater(len(calls) * 8 / total, 0.5)


class PatchLifecycleTest(unittest.TestCase):
    def test_patches_are_applied_and_restored(self):
        original_forward = modeling_qwen3.Qwen3RMSNorm.forward
        original_rope = modeling_qwen3.apply_rotary_pos_emb
        with fused_qwen3_modules() as applied:
            self.assertEqual(applied, {"rms_norm": True, "rope": True})
            self.assertIsNot(modeling_qwen3.Qwen3RMSNorm.forward, original_forward)
            self.assertIs(modeling_qwen3.apply_rotary_pos_emb, fused_apply_rotary_pos_emb)
        self.assertIs(modeling_qwen3.Qwen3RMSNorm.forward, original_forward)
        self.assertIs(modeling_qwen3.apply_rotary_pos_emb, original_rope)

    def test_selective_patching_and_restore_on_error(self):
        original_forward = modeling_qwen3.Qwen3RMSNorm.forward
        original_rope = modeling_qwen3.apply_rotary_pos_emb
        with fused_qwen3_modules(rms_norm=False) as applied:
            self.assertEqual(applied, {"rms_norm": False, "rope": True})
            self.assertIs(modeling_qwen3.Qwen3RMSNorm.forward, original_forward)
            self.assertIs(modeling_qwen3.apply_rotary_pos_emb, fused_apply_rotary_pos_emb)
        with self.assertRaises(RuntimeError):
            with fused_qwen3_modules():
                raise RuntimeError("boom")
        self.assertIs(modeling_qwen3.Qwen3RMSNorm.forward, original_forward)
        self.assertIs(modeling_qwen3.apply_rotary_pos_emb, original_rope)


if __name__ == "__main__":
    unittest.main()
