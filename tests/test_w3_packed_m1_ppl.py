import importlib.util
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_qwen3_8b_w3_packed_m1_ppl",
    ROOT / "scripts/run_qwen3_8b_w3_packed_m1_ppl.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class TinyCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length


class TinyCachedLM(torch.nn.Module):
    def __init__(self, vocab_size=7):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1):
        cache = TinyCache() if past_key_values is None else past_key_values
        cache.length += input_ids.shape[1]
        token = input_ids[:, -1]
        logits = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        logits = logits + token.view(1, 1, 1) * 0.01
        return SimpleNamespace(logits=logits, past_key_values=cache)


class PackedM1PPLTests(unittest.TestCase):
    def test_frozen_protocol_has_no_guessed_quality_threshold(self):
        config = json.loads(
            (ROOT / "configs/evaluation/qwen3_8b_w3_packed_m1_ppl_v1.json").read_text()
        )
        RUNNER.validate_config(config)
        self.assertIsNone(config["acceptance"]["quality_effect_threshold"])
        self.assertEqual(config["accepted_protocol"]["scored_transition_count"], 298862)
        self.assertEqual(config["evaluation"]["tokens_per_model_call"], 1)
        self.assertEqual(
            config["evaluation"]["fused_nonlinear_modules"],
            {"rms_norm": True, "rope": True},
        )

    def test_cpu_scorer_matches_teacher_forced_cross_entropy(self):
        blocks = [[0, 1, 2], [3, 4, 5]]
        model = TinyCachedLM().eval()
        result = RUNNER.score_m1_ppl(
            model,
            blocks,
            arm="original_bf16",
            device=torch.device("cpu"),
            expected_packed_linears=0,
        )
        logits = torch.arange(7, dtype=torch.float32).view(1, -1)
        labels = torch.tensor([1, 2, 4, 5])
        expected_nll = float(
            torch.nn.functional.cross_entropy(
                logits.expand(labels.numel(), -1), labels, reduction="sum"
            )
        )
        self.assertEqual(result["scored_transition_count"], 4)
        self.assertEqual(result["block_count"], 2)
        self.assertAlmostEqual(result["total_nll"], expected_nll, places=5)
        self.assertAlmostEqual(result["perplexity"], math.exp(expected_nll / 4), places=5)
        self.assertIsNone(result["packed_route_smoke"])

    def test_comparison_reports_effect_without_accepting_it(self):
        baseline = {"arm": "decoded", "perplexity": 10.0, "mean_nll": math.log(10.0)}
        candidate = {"arm": "packed", "perplexity": 10.5, "mean_nll": math.log(10.5)}
        result = RUNNER.comparison(candidate, baseline)
        self.assertAlmostEqual(result["ppl_delta"], 0.5)
        self.assertAlmostEqual(result["ppl_relative_percent"], 5.0)
        self.assertNotIn("passed", result)


if __name__ == "__main__":
    unittest.main()
