import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_w3_lut_benchmark as runner
sys.path.pop(0)


class W3LUTConfigTest(unittest.TestCase):
    def test_frozen_four_by_three_contract(self):
        config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        runner.validate_config(config)
        self.assertEqual(len(config["shape_classes"]), 4)
        self.assertEqual(len(config["row_tiles"]), 3)
        self.assertEqual(config["candidate_count"], 12)
        self.assertEqual(config["mode"], "cuda_graph_total")
        self.assertNotIn("prepare", config.get("baselines", []))

    def test_shape_and_prepare_policy_are_fail_closed(self):
        config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["expected_shapes"]["k_v"] = [2048, 4096]
        with self.assertRaises(ValueError):
            runner.validate_config(changed)
        changed = copy.deepcopy(config)
        changed["prepare_policy"] = "benchmark prepare now"
        with self.assertRaises(ValueError):
            runner.validate_config(changed)


if __name__ == "__main__":
    unittest.main()
