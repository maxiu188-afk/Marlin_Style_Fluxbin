import importlib.util
import sys
import unittest
from pathlib import Path


def load_runner():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_qwen3_full_hessian_obq_s8_ppl.py"
        spec = importlib.util.spec_from_file_location("full_hessian_obq_ppl", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class FullHessianOBQPPLTests(unittest.TestCase):
    def test_independent_arm_payload_inventories(self) -> None:
        runner = load_runner()
        pure = runner.expected_payload_keys("pure")
        hybrid = runner.expected_payload_keys("hybrid_s8")
        self.assertEqual(len(pure), 21)
        self.assertEqual(len(hybrid), 49)
        self.assertTrue(pure < hybrid)
        self.assertIn("mlp.down_proj.refinement_indices", hybrid)
        with self.assertRaises(ValueError):
            runner.expected_payload_keys("shared")

    def test_matched_target_validation_rejects_arm_drift(self) -> None:
        runner = load_runner()
        pure = [
            {
                "layer_index": 0,
                "metadata": {
                    "linears": [
                        {
                            "module": "self_attn.q_proj",
                            "shape": [2, 2],
                            "parameter_count": 4,
                            "target_bf16_sha256": "same",
                        }
                    ]
                },
            }
        ]
        hybrid = [
            {
                "layer_index": 0,
                "metadata": {
                    "linears": [
                        {
                            "module": "self_attn.q_proj",
                            "shape": [2, 2],
                            "parameter_count": 4,
                            "target_bf16_sha256": "same",
                        }
                    ]
                },
            }
        ]
        runner.validate_matched_targets(pure, hybrid)
        hybrid[0]["metadata"]["linears"][0]["target_bf16_sha256"] = "drift"
        with self.assertRaises(ValueError):
            runner.validate_matched_targets(pure, hybrid)


if __name__ == "__main__":
    unittest.main()
