import unittest

import torch
from transformers import Qwen3Config
from transformers.models.qwen3 import modeling_qwen3

from fluxbin_style.offline_rotation import (
    QWEN3_ROTATION_FORMAT,
    apply_offline_qwen3_rotation,
    assert_standard_qwen3_layout,
    normalized_hadamard_matrix,
)


def build_tiny_qwen3(*, tie: bool = False) -> modeling_qwen3.Qwen3ForCausalLM:
    """A grouped-query Qwen3 with a non-power-of-two intermediate size."""

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        tie_word_embeddings=tie,
    )
    torch.manual_seed(20260917)
    model = modeling_qwen3.Qwen3ForCausalLM(config).to(torch.float32).eval()
    # Random init gives every norm weight ~1; make the fusion step observable.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
            layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
            layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)
        model.model.norm.weight.uniform_(0.5, 1.5)
    return model


TOKENS = torch.tensor([[7, 19, 3, 200, 41, 8, 255, 62]])


def logits_of(model: torch.nn.Module) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids=TOKENS, use_cache=False).logits.clone()


class OfflineRotationTest(unittest.TestCase):
    def test_hadamard_is_symmetric_and_orthogonal(self) -> None:
        for size in (16, 128, 4096):
            h = normalized_hadamard_matrix(size, torch.float64, torch.device("cpu"))
            self.assertTrue(torch.equal(h, h.T))
            identity = torch.eye(size, dtype=torch.float64)
            self.assertLess((h @ h - identity).abs().max().item(), 1e-12)
        with self.assertRaises(ValueError):
            normalized_hadamard_matrix(12288, torch.float32, torch.device("cpu"))

    def test_rotation_preserves_logits(self) -> None:
        model = build_tiny_qwen3()
        before = logits_of(model)
        record = apply_offline_qwen3_rotation(model)
        after = logits_of(model)
        self.assertEqual(record["format"], QWEN3_ROTATION_FORMAT)
        self.assertFalse(record["down_proj_input_rotated"])
        self.assertFalse(record["custom_kernel"])
        scale = before.abs().max().item()
        self.assertGreater(scale, 0.0)
        self.assertLess((after - before).abs().max().item() / scale, 1e-4)

    def test_rotation_without_value_rotation_preserves_logits(self) -> None:
        model = build_tiny_qwen3()
        before = logits_of(model)
        record = apply_offline_qwen3_rotation(model, rotate_values=False)
        after = logits_of(model)
        self.assertFalse(record["rotate_values"])
        self.assertLess(
            (after - before).abs().max().item() / before.abs().max().item(), 1e-4
        )

    def test_rotation_changes_every_quantization_target(self) -> None:
        model = build_tiny_qwen3()
        original = {name: value.clone() for name, value in model.named_parameters()}
        apply_offline_qwen3_rotation(model)
        for layer_index in range(len(model.model.layers)):
            prefix = f"model.layers.{layer_index}."
            for name in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ):
                key = f"{prefix}{name}.weight"
                current = dict(model.named_parameters())[key]
                self.assertEqual(current.shape, original[key].shape)
                self.assertFalse(torch.equal(current, original[key]), key)

    def test_value_rotation_touches_only_the_value_output_path(self) -> None:
        with_r2 = build_tiny_qwen3()
        apply_offline_qwen3_rotation(with_r2, rotate_values=True)
        without_r2 = build_tiny_qwen3()
        apply_offline_qwen3_rotation(without_r2, rotate_values=False)
        for name, expected_change in (
            ("self_attn.v_proj", True),
            ("self_attn.o_proj", True),
            ("self_attn.q_proj", False),
            ("self_attn.k_proj", False),
            ("mlp.down_proj", False),
        ):
            left = with_r2.model.layers[0].get_submodule(name).weight
            right = without_r2.model.layers[0].get_submodule(name).weight
            self.assertEqual(not torch.equal(left, right), expected_change, name)

    def test_norms_become_unit_and_head_norms_are_untouched(self) -> None:
        model = build_tiny_qwen3()
        head_norms = {
            name: value.clone()
            for name, value in model.named_parameters()
            if "q_norm" in name or "k_norm" in name
        }
        apply_offline_qwen3_rotation(model)
        for layer in model.model.layers:
            self.assertTrue(torch.equal(layer.input_layernorm.weight, torch.ones(128)))
            self.assertTrue(
                torch.equal(layer.post_attention_layernorm.weight, torch.ones(128))
            )
        self.assertTrue(torch.equal(model.model.norm.weight, torch.ones(128)))
        current = dict(model.named_parameters())
        for name, value in head_norms.items():
            self.assertTrue(torch.equal(current[name], value), name)

    def test_tied_embeddings_are_untied_and_logits_preserved(self) -> None:
        model = build_tiny_qwen3(tie=True)
        before = logits_of(model)
        record = apply_offline_qwen3_rotation(model)
        self.assertTrue(record["output_head_was_untied"])
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertNotEqual(
            model.model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )
        after = logits_of(model)
        self.assertLess(
            (after - before).abs().max().item() / before.abs().max().item(), 1e-4
        )

    def test_layout_gate_rejects_other_architectures(self) -> None:
        model = build_tiny_qwen3()
        model.config.model_type = "llama"
        with self.assertRaisesRegex(ValueError, "qwen3"):
            assert_standard_qwen3_layout(model)


if __name__ == "__main__":
    unittest.main()
