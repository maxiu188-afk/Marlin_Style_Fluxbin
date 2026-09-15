import json
import os
import sys
import unittest
from pathlib import Path

import torch

from fluxbin_style import QWEN3_LINEAR_MODULES, tensor_sha256
from fluxbin_style.rate_distortion import (
    EXPECTED_WEIGHTS,
    analytical_gptq_storage,
    analytical_qbb_storage,
    build_summary_rows,
    classify_gptq_tensor,
    render_summary_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json"
sys.path.insert(0, str(ROOT / "scripts"))
import convert_qwen3_8b_qbb_fp16_scales as fp16_converter
import run_qwen3_8b_w3_gptq as gptq_runner
import run_qwen3_8b_w3_rate_distortion_ppl as ppl_runner
sys.path.pop(0)


class RateDistortionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_frozen_primary_gptq_contract(self):
        gptq_runner.validate_static_config(self.config)
        changed = json.loads(CONFIG.read_text())
        changed["gptq"]["desc_act"] = False
        with self.assertRaises(ValueError):
            gptq_runner.validate_static_config(changed)
        changed = json.loads(CONFIG.read_text())
        changed["gptq"]["pack_impl"] = "gpu"
        with self.assertRaises(ValueError):
            gptq_runner.validate_static_config(changed)

    def test_exact_8b_qbb_and_gptq_analytical_storage(self):
        qbb32 = analytical_qbb_storage(self.config["model"], scale_bytes=4)
        qbb16 = analytical_qbb_storage(self.config["model"], scale_bytes=2)
        gptq = analytical_gptq_storage(self.config["model"])
        self.assertEqual(qbb32["quantized_weight_count"], EXPECTED_WEIGHTS)
        self.assertEqual(qbb32["persistent_tensor_bytes"], 2_724_636_672)
        self.assertAlmostEqual(qbb32["effective_bits_per_weight"], 3.13818359375)
        self.assertEqual(qbb16["persistent_tensor_bytes"], 2_284_886_016)
        self.assertAlmostEqual(qbb16["effective_bits_per_weight"], 2.6316873301630435)
        self.assertEqual(gptq["persistent_tensor_bytes"], 2_713_190_400)
        self.assertEqual(gptq["effective_bits_per_weight"], 3.125)
        self.assertFalse(qbb32["includes_lookup"])

    def test_fp16_scale_conversion_preserves_every_fixed_tensor(self):
        tensors = {}
        for index, module in enumerate(QWEN3_LINEAR_MODULES):
            tensors[f"{module}.global_sign_codes"] = torch.tensor([[index]], dtype=torch.uint8)
            tensors[f"{module}.refinement_indices"] = torch.tensor([[index]], dtype=torch.int16)
            tensors[f"{module}.refinement_sign_codes"] = torch.tensor([[index + 1]], dtype=torch.uint8)
            for suffix in (
                "global_row_scales",
                "global_column_scales",
                "refinement_row_scales",
                "refinement_column_scales",
            ):
                tensors[f"{module}.{suffix}"] = torch.tensor([index + 0.25], dtype=torch.float32)
        fixed_before = {
            name: tensor_sha256(value)
            for name, value in tensors.items()
            if not name.endswith("_scales")
        }
        converted, record = fp16_converter.convert_tensors(tensors)
        self.assertEqual(set(converted), fp16_converter.expected_keys())
        self.assertEqual(record["fixed_tensor_sha256"], fixed_before)
        for name, value in converted.items():
            if name.endswith("_scales"):
                self.assertEqual(value.dtype, torch.float16)
            else:
                self.assertEqual(tensor_sha256(value), fixed_before[name])

    def test_fp16_scale_source_contract_rejects_layout_drift_before_io(self):
        changed = json.loads(CONFIG.read_text())
        changed["qbb"]["columns_per_group"] = 16
        args = type("Args", (), {})()
        with self.assertRaises(ValueError):
            fp16_converter.validate_source(changed, args)

    def test_gptq_storage_categories_are_explicit(self):
        self.assertEqual(classify_gptq_tensor("model.layers.0.mlp.up_proj.qweight"), "packed_weight_codes")
        self.assertEqual(classify_gptq_tensor("model.layers.0.mlp.up_proj.qzeros"), "zero_points")
        self.assertEqual(classify_gptq_tensor("model.layers.0.mlp.up_proj.g_idx"), "group_indices_or_permutation")
        self.assertIsNone(classify_gptq_tensor("model.embed_tokens.weight"))

    def test_gptq_decode_forces_eager_torch_unpack(self):
        previous = os.environ.get("GPTQ_TORCH_TRITON_DEQUANT")
        try:
            os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "1"
            gptq_runner.force_eager_torch_dequantizer()
            self.assertEqual(os.environ["GPTQ_TORCH_TRITON_DEQUANT"], "0")
        finally:
            if previous is None:
                os.environ.pop("GPTQ_TORCH_TRITON_DEQUANT", None)
            else:
                os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = previous

    def test_summary_has_requested_deltas_and_no_automatic_decision(self):
        arms = {
            name: {
                "perplexity": ppl,
                "mean_nll": 1.0,
                "metrics_valid": True,
                "scored_transition_count": 298862,
            }
            for name, ppl in (
                ("bf16", 9.725),
                ("qbb_current", 13.17),
                ("gptq_w3_g128_sym", 11.5),
                ("qbb_fp16_scales", 13.2),
            )
        }
        storage = {
            name: {
                "nominal_bits": 3,
                "effective_bits_per_weight": 3.125,
                "serialized_weight_storage_bytes": 10,
                "linear_coverage": "252/252",
            }
            for name in arms
        }
        rows = build_summary_rows(arms, storage)
        self.assertAlmostEqual(rows[2]["delta_ppl_vs_current_qbb"], -1.67)
        markdown = render_summary_markdown(rows, status="completed_pending_effect_size_review")
        self.assertIn("298,862", markdown)
        self.assertIn("no small", markdown)

    def test_bf16_and_current_qbb_are_reproduction_gates(self):
        ppl_runner.validate_evaluation_config(self.config)
        bf16 = {
            "perplexity": self.config["evaluation"]["accepted_reproduction_reference"]["bf16_perplexity"]
        }
        self.assertTrue(ppl_runner.validate_reference_reproduction(self.config, "bf16", bf16)["passed"])
        bad = {"perplexity": bf16["perplexity"] + 0.01}
        with self.assertRaises(RuntimeError):
            ppl_runner.validate_reference_reproduction(self.config, "bf16", bad)
        self.assertFalse(
            ppl_runner.validate_reference_reproduction(self.config, "gptq_w3_g128_sym", {})["required"]
        )
        changed = json.loads(CONFIG.read_text())
        changed["evaluation"]["logit_chunk_tokens"] = 256
        with self.assertRaises(ValueError):
            ppl_runner.validate_evaluation_config(changed)


if __name__ == "__main__":
    unittest.main()
