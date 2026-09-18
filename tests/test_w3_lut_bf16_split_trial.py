import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_w3_lut_bf16_split_trial.py"
SPEC = importlib.util.spec_from_file_location("w3_bf16_split_trial", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W3Bf16SplitTrialTest(unittest.TestCase):
    def test_bounded_candidate_contract(self):
        config = json.loads(RUNNER.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        RUNNER.validate_config(config)
        self.assertEqual(RUNNER.candidate_count(config), 46)
        self.assertEqual(config["arithmetic_modes"], ["structural", "decoded_bf16"])
        self.assertIn(2048, config["shape_classes"]["gate_up"]["row_tiles"])
        self.assertIn(2048, config["shape_classes"]["down"]["row_tiles"])

    def test_rejects_scope_and_candidate_drift(self):
        config = json.loads(RUNNER.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["candidate_count"] += 1
        with self.assertRaisesRegex(ValueError, "contract drifted"):
            RUNNER.validate_config(changed)
        changed = copy.deepcopy(config)
        changed["shape_classes"]["q_o"]["groups_per_split"] = [0]
        changed["candidate_count"] = RUNNER.candidate_count(changed)
        with self.assertRaisesRegex(ValueError, "groups_per_split"):
            RUNNER.validate_config(changed)


if __name__ == "__main__":
    unittest.main()
