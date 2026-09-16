import unittest

import torch
from torch import nn

from fluxbin_style.full_model_trial import packed_modules
from fluxbin_style.full_model_trial import compare_trace, decode_trace
from fluxbin_style.gptq_deployment import (
    convert_gptq_w3_to_planar,
    restore_planar_w3,
)
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES
from fluxbin_style.w3_lut_deployment import (
    PackedW3Linear,
    qwen3_row_tile,
    replace_w3_block_linears,
)
from fluxbin_style.static_decode import StaticDecodeSession, prepared_linears
from test_gptq_w3_planar import synthetic_raw


def tiny_layout(out_features=32, in_features=128):
    raw, _, _ = synthetic_raw(k=in_features, o=out_features)
    return convert_gptq_w3_to_planar(raw, qzero_format=1)


def tiny_block():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.mlp = nn.Module()
    for name in QWEN3_LINEAR_MODULES:
        parent, attribute = name.rsplit(".", 1)
        setattr(block.get_submodule(parent), attribute, nn.Linear(128, 32, bias=False))
    return block


class PackedW3LinearTest(unittest.TestCase):
    def test_dense_fallback_matches_restored_weight(self):
        layout = tiny_layout()
        module = PackedW3Linear(layout, fallback="dense", row_tile=256)
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        with torch.inference_mode():
            actual = module(x)
            expected = torch.nn.functional.linear(x, restore_planar_w3(layout))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(module.route_counts, {"packed_m1": 0, "dense_fallback": 1})

    def test_strict_cpu_and_prepared_binding_fail_closed(self):
        module = PackedW3Linear(tiny_layout(), row_tile=256)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "unsupported input"):
            module(torch.randn(1, 128, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "prepared decode requires CUDA"):
            module.bind_prepared_decode(torch.bfloat16)

    def test_shape_policy_is_frozen(self):
        self.assertEqual(qwen3_row_tile(4096, 4096), 256)
        self.assertEqual(qwen3_row_tile(1024, 4096), 512)
        self.assertEqual(qwen3_row_tile(12288, 4096), 1024)
        self.assertEqual(qwen3_row_tile(4096, 12288), 1024)
        with self.assertRaises(ValueError):
            qwen3_row_tile(32, 128)

    def test_complete_block_replacement_and_discovery(self):
        layout = tiny_layout()
        payload = {
            f"{name}.{field}": value.clone()
            for name in QWEN3_LINEAR_MODULES
            for field, value in layout.items()
        }
        block = tiny_block()
        names = replace_w3_block_linears(
            block, payload, fallback="dense", row_tile_by_shape={(32, 128): 256}
        )
        self.assertEqual(names, list(QWEN3_LINEAR_MODULES))
        self.assertEqual(len(packed_modules(block)), 7)
        self.assertTrue(all(isinstance(module, PackedW3Linear) for module in packed_modules(block).values()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires NVIDIA CUDA compiler/device")
    def test_tiny_qwen_static_graph_routes_all_w3_linears(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16).eval()
        for layer in model.model.layers:
            for name in QWEN3_LINEAR_MODULES:
                linear = layer.get_submodule(name)
                layout = tiny_layout(linear.out_features, linear.in_features)
                parent, attribute = name.rsplit(".", 1)
                setattr(
                    layer.get_submodule(parent),
                    attribute,
                    PackedW3Linear(layout, fallback="dense", row_tile=256).cuda(),
                )
        input_ids = torch.tensor([[1, 3, 7]], device="cuda")
        dynamic = decode_trace(model, input_ids, steps=3)
        session = StaticDecodeSession(model, input_ids, dynamic["fed_tokens"])
        with prepared_linears(model, torch.bfloat16):
            expected = session.audit()
            self.assertTrue(compare_trace(expected, dynamic, logprob_tolerance=0.05)["passed"])
            self.assertEqual(len(expected["routes"]), 14)
            self.assertTrue(
                all(
                    route == {"dense_fallback": 0, "packed_m1": 3}
                    for route in expected["routes"].values()
                )
            )
            graph = session.capture()
            self.assertTrue(torch.equal(expected["logits"], graph["logits"]))


if __name__ == "__main__":
    unittest.main()
