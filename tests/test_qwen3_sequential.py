import unittest

import torch

from fluxbin_style import (
    capture_first_layer_inputs,
    capture_layer_hessians,
    propagate_layer_inputs,
)


class Qwen3SequentialTests(unittest.TestCase):
    def test_tiny_qwen3_capture_and_propagation(self) -> None:
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("transformers is unavailable")
        torch.manual_seed(47)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
        model = Qwen3ForCausalLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        capture = capture_first_layer_inputs(
            model,
            tokens,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(capture.inputs.shape), (2, 8, 16))
        layer = model.model.layers[0]
        hessians = capture_layer_hessians(
            layer,
            capture.inputs,
            capture.forward_kwargs,
        )
        self.assertEqual(set(hessians.hessians), {"qkv", "o", "gate_up", "down"})
        self.assertEqual(tuple(hessians.hessians["qkv"].shape), (16, 16))
        self.assertEqual(tuple(hessians.hessians["o"].shape), (16, 16))
        self.assertEqual(tuple(hessians.hessians["gate_up"].shape), (16, 16))
        self.assertEqual(tuple(hessians.hessians["down"].shape), (32, 32))
        self.assertEqual(set(hessians.activation_rows.values()), {16})
        outputs = torch.empty_like(capture.inputs)
        propagate_layer_inputs(
            layer,
            capture.inputs,
            outputs,
            capture.forward_kwargs,
        )
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertFalse(torch.equal(outputs, capture.inputs))


if __name__ == "__main__":
    unittest.main()
