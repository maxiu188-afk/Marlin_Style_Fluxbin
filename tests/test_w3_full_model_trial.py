import importlib.util
import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_qwen3_8b_w3_full_m1_trial.py"
SPEC = importlib.util.spec_from_file_location("w3_full_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W3FullModelTrialTest(unittest.TestCase):
    def test_protocol_preserves_prompts_and_freezes_graph_primary(self):
        config = json.loads(RUNNER.PROTOCOL.read_text(encoding="utf-8"))
        previous = json.loads(
            (ROOT / "configs/acceleration/qwen3_8b_full_m1_v2.json").read_text(encoding="utf-8")
        )
        RUNNER.validate_protocol(config)
        for key in ("seed", "decode_steps", "warmup", "repeats", "prompts", "max_prompt_tokens"):
            self.assertEqual(config[key], previous[key])
        self.assertEqual(config["primary_mode"], "sequence_graph")
        self.assertFalse(config["prepare_candidate"])

    def test_timing_and_exact_route_guards(self):
        self.assertTrue(RUNNER.statistics_row([10.0, 10.1], 0.05)["stable"])
        self.assertFalse(RUNNER.statistics_row([10.0, 12.0], 0.05)["stable"])
        for samples in ([], [float("nan")], [0.0], [-1.0]):
            with self.assertRaises(ValueError):
                RUNNER.statistics_row(samples, 0.05)
        routes = {str(index): {"dense_fallback": 0, "packed_m1": 32} for index in range(252)}
        RUNNER.require_routes(routes, 32)
        routes["0"]["dense_fallback"] = 1
        with self.assertRaises(RuntimeError):
            RUNNER.require_routes(routes, 32)
        trace = {
            "cache_length": 5,
            "logits": torch.ones(1),
            "predictions": torch.ones(1),
            "fed_tokens": torch.ones(1),
        }
        RUNNER.exact_trace(trace, trace)
        with self.assertRaises(RuntimeError):
            RUNNER.exact_trace({**trace, "logits": torch.zeros(1)}, trace)


if __name__ == "__main__":
    unittest.main()
