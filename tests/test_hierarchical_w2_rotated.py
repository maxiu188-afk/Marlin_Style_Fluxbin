import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from fluxbin_style import QWEN3_LINEAR_MODULES
from fluxbin_style.offline_rotation import QWEN3_ROTATION_FORMAT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_qwen3_8b_hierarchical_w2_gptq import (  # noqa: E402
    SOURCE_FILES as UNROTATED_SOURCE_FILES,
)
from run_qwen3_8b_hierarchical_w2_rotated_gptq import (  # noqa: E402
    ARM,
    SOURCE_FILES as ROTATED_SOURCE_FILES,
    bind_runtime_identity,
    rotate_model,
    runtime_identity,
    source_hash,
    validate_config,
)
from run_qwen3_8b_hierarchical_w2_rotation_probe import (  # noqa: E402
    ARMS as PROBE_ARMS,
    PROBE_SOURCE_FILES,
    evaluate_gate,
    source_hash as probe_source_hash,
    validate_probe_config,
)
sys.path.pop(0)

sys.path.insert(0, str(ROOT / "tests"))
from test_offline_rotation import build_tiny_qwen3, logits_of  # noqa: E402
sys.path.pop(0)

CONFIG_PATH = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
UNROTATED_CONFIG_PATH = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


class RotatedConfigTest(unittest.TestCase):
    def test_frozen_config_passes_its_own_gate(self) -> None:
        validate_config(load_config())

    def test_single_arm_at_the_unrotated_budget(self) -> None:
        config = load_config()
        unrotated = json.loads(UNROTATED_CONFIG_PATH.read_text())
        self.assertEqual(list(config["variants"]), [ARM])
        self.assertEqual(config["variants"][ARM], unrotated["variants"][ARM])
        self.assertEqual(config["variants"][ARM]["nominal_bpw"], 2.5)

    def test_quantization_controls_match_the_unrotated_baseline(self) -> None:
        config = load_config()
        unrotated = json.loads(UNROTATED_CONFIG_PATH.read_text())
        # Only the rotation may differ; otherwise the comparison is confounded.
        for section in ("model", "calibration", "gptq", "hierarchical_projection"):
            self.assertEqual(config[section], unrotated[section], section)
        for key in ("accepted_protocol", "attention_implementation", "logit_chunk_tokens"):
            self.assertEqual(
                config["evaluation"][key], unrotated["evaluation"][key], key
            )

    def test_rotation_block_excludes_kernels_and_online_transforms(self) -> None:
        rotation = load_config()["rotation"]
        self.assertEqual(rotation["format"], QWEN3_ROTATION_FORMAT)
        for key in (
            "online_mlp_hadamard",
            "online_qk_hadamard",
            "down_proj_input_rotated",
            "activation_quantization",
            "custom_kernel",
            "learned_rotation",
        ):
            self.assertFalse(rotation[key], key)
        self.assertTrue(rotation["rotate_values"])

    def test_policy_is_all_true_and_names_the_route(self) -> None:
        policy = load_config()["policy"]
        self.assertTrue(all(policy.values()))
        self.assertTrue(policy["offline_rotation_only"])
        self.assertTrue(policy["no_cuda_kernel"])
        self.assertTrue(policy["no_online_hadamard"])
        # The unrotated config's blanket no_rotation flag must not reappear here.
        self.assertNotIn("no_rotation", policy)

    def test_recorded_baselines_match_the_accepted_endpoint(self) -> None:
        baselines = load_config()["baselines"]
        self.assertEqual(baselines["unrotated_h2_50_perplexity"], 27.932835)
        self.assertEqual(baselines["w3_endpoint_same_run_perplexity"], 11.266808)
        self.assertEqual(baselines["bf16_reference_perplexity"], 9.724945)

    def test_gate_rejects_drift(self) -> None:
        cases = {
            "online hadamard": lambda c: c["rotation"].__setitem__("online_mlp_hadamard", True),
            "kernel": lambda c: c["rotation"].__setitem__("custom_kernel", True),
            "learned rotation": lambda c: c["rotation"].__setitem__("learned_rotation", True),
            "second arm": lambda c: c["variants"].__setitem__(
                "H2.875", {"r32_bits": 8, "r16_bits": 8, "nominal_bpw": 2.875}
            ),
            "gptq control": lambda c: c["gptq"].__setitem__("desc_act", False),
            "disabled policy": lambda c: c["policy"].__setitem__("no_cuda_kernel", False),
            "stale baseline hash": lambda c: c["baselines"].__setitem__(
                "unrotated_source_config_sha256", "0" * 64
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                config = load_config()
                mutate(config)
                with self.assertRaises(ValueError):
                    validate_config(config)


class RotatedRunnerTest(unittest.TestCase):
    def test_frozen_unrotated_implementation_hash_is_not_disturbed(self) -> None:
        # Adding the rotation route must not invalidate the accepted
        # H2.50/H2.875 artifacts, whose implementation_sha256 covers only these.
        for name in ("src/fluxbin_style/offline_rotation.py",
                     "scripts/run_qwen3_8b_hierarchical_w2_rotated_gptq.py"):
            self.assertNotIn(name, UNROTATED_SOURCE_FILES)
        self.assertIn("src/fluxbin_style/offline_rotation.py", ROTATED_SOURCE_FILES)
        for name in UNROTATED_SOURCE_FILES:
            self.assertIn(name, ROTATED_SOURCE_FILES)

    def test_source_hash_covers_every_listed_file(self) -> None:
        digest, files = source_hash()
        self.assertEqual(set(files), set(ROTATED_SOURCE_FILES))
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, source_hash()[0])

    def test_runtime_identity_rejects_cross_device_resume(self) -> None:
        formal = {
            "execution_policy": "formal-a100",
            "formal_a100_device_match": True,
            "device": "NVIDIA A100 80GB PCIe",
            "compute_capability": [8, 0],
            "python": "3.11.9",
            "torch": "2.7.1+cu128",
            "transformers": "4.53.2",
            "datasets": "4.0.0",
            "safetensors": "0.5.3",
            "cuda": "12.8",
        }
        cross_device = {
            **formal,
            "execution_policy": "same-device-quality",
            "formal_a100_device_match": False,
            "device": "NVIDIA RTX PRO 4500 Blackwell Generation",
            "compute_capability": [12, 0],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "layers"
            bind_runtime_identity(artifact_dir, runtime_identity(formal))
            with self.assertRaisesRegex(RuntimeError, "different runtime identity"):
                bind_runtime_identity(artifact_dir, runtime_identity(cross_device))

    def test_rotate_model_applies_and_cross_checks_the_config(self) -> None:
        config = load_config()
        config["rotation"] = {
            **config["rotation"], "residual_size": 128, "head_rotation_size": 16
        }
        model = build_tiny_qwen3()
        before = logits_of(model)
        record = rotate_model(model, config, device=torch.device("cpu"))
        after = logits_of(model)
        self.assertEqual(record["format"], QWEN3_ROTATION_FORMAT)
        self.assertLess(
            (after - before).abs().max().item() / before.abs().max().item(), 1e-4
        )

    def test_rotate_model_rejects_a_config_it_did_not_apply(self) -> None:
        config = load_config()
        config["rotation"] = {
            **config["rotation"],
            "residual_size": 128,
            "head_rotation_size": 16,
            "down_proj_input_rotated": True,
        }
        with self.assertRaisesRegex(RuntimeError, "down_proj_input_rotated"):
            rotate_model(build_tiny_qwen3(), config, device=torch.device("cpu"))

    def test_rotate_model_rejects_a_geometry_mismatch(self) -> None:
        config = copy.deepcopy(load_config())
        with self.assertRaisesRegex(RuntimeError, "residual_size"):
            rotate_model(build_tiny_qwen3(), config, device=torch.device("cpu"))


class RotatedProbeTest(unittest.TestCase):
    @staticmethod
    def arms(layers: list[int], ratios: list[float]) -> dict:
        arms = {name: {"layers": []} for name in PROBE_ARMS}
        for layer, ratio in zip(layers, ratios, strict=True):
            original_linears = []
            rotated_linears = []
            for module in QWEN3_LINEAR_MODULES:
                original_linears.append(
                    {"module": module, "squared_error": 10.0, "target_squared_norm": 100.0}
                )
                rotated_linears.append(
                    {
                        "module": module,
                        "squared_error": 10.0 * ratio,
                        "target_squared_norm": 100.0,
                    }
                )
            arms[PROBE_ARMS[0]]["layers"].append(
                {
                    "layer_index": layer,
                    "block_output": {
                        "squared_error": 10.0,
                        "reference_squared_norm": 100.0,
                        "relative_squared_error": 0.1,
                    },
                    "linears": original_linears,
                }
            )
            arms[PROBE_ARMS[1]]["layers"].append(
                {
                    "layer_index": layer,
                    "block_output": {
                        "squared_error": 10.0 * ratio,
                        "reference_squared_norm": 100.0,
                        "relative_squared_error": 0.1 * ratio,
                    },
                    "linears": rotated_linears,
                }
            )
        return arms

    def test_probe_config_and_source_hash_are_frozen(self) -> None:
        config = load_config()
        validate_probe_config(config)
        digest, files = probe_source_hash()
        self.assertEqual(set(files), set(PROBE_SOURCE_FILES))
        self.assertEqual(len(digest), 64)
        drifted = copy.deepcopy(config)
        drifted["probe"]["full_model_auto_launch"] = True
        with self.assertRaises(ValueError):
            validate_probe_config(drifted)

    def test_layer0_gate_is_only_a_permissive_rejection_gate(self) -> None:
        config = load_config()
        passed = evaluate_gate(config, "layer0", self.arms([0], [0.9]))
        stopped = evaluate_gate(config, "layer0", self.arms([0], [0.99]))
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["action"], "advance_to_representative_probe")
        self.assertFalse(stopped["passed"])
        self.assertEqual(stopped["action"], "stop_without_full_model")

    def test_representative_gate_requires_strong_cross_depth_gain(self) -> None:
        config = load_config()
        passed = evaluate_gate(
            config, "representative", self.arms([0, 17, 35], [0.8, 0.8, 0.8])
        )
        stopped = evaluate_gate(
            config, "representative", self.arms([0, 17, 35], [0.8, 0.8, 1.2])
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(
            passed["action"], "eligible_for_manual_full_model_launch"
        )
        self.assertFalse(stopped["passed"])

    def test_representative_stage_reuses_layer0_instead_of_recomputing_it(self) -> None:
        stage = load_config()["probe"]["stages"]["representative"]
        self.assertEqual(stage["execution_layers"], [17, 35])
        self.assertEqual(stage["evaluation_layers"], [0, 17, 35])

    def test_probe_wrapper_cannot_launch_full_model_or_ppl(self) -> None:
        source = (
            ROOT / "scripts/run_qwen3_8b_hierarchical_w2_rotation_probe_job.sh"
        ).read_text()
        self.assertNotIn("run_qwen3_8b_hierarchical_w2_rotated_job.sh", source)
        self.assertNotIn("run_qwen3_8b_hierarchical_w2_rotated_ppl.py", source)
        self.assertIn("completed_probe_passed_pending_manual_full_model", source)

class RotatedPplGateTest(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_qwen3_8b_hierarchical_w2_rotated_ppl as module

        sys.path.pop(0)
        self.module = module
        self.config = load_config()
        self.config_hash = "a" * 64
        self.implementation_hash = "b" * 64

    def passing_result(self) -> dict:
        return {
            "status": "completed_pending_ppl",
            "arm": ARM,
            "experiment_id": self.config["experiment_id"],
            "config_sha256": self.config_hash,
            "implementation_sha256": self.implementation_hash,
            "variant": self.config["variants"][ARM],
            "rotation": {"format": self.config["rotation"]["format"]},
            "execution": {"execution_policy": "formal-a100"},
        }

    def full_execution(self) -> dict:
        return {
            "execution_policy": "formal-a100",
            "formal_a100_device_match": True,
            "device": "NVIDIA A100 80GB PCIe",
            "compute_capability": [8, 0],
            "python": "3.11.9",
            "torch": "2.7.1+cu128",
            "transformers": "4.53.2",
            "datasets": "4.0.0",
            "safetensors": "0.5.3",
            "cuda": "12.8",
        }

    def check(self, result: dict) -> None:
        path = ROOT / "tmp" / "rotated-result-fixture.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        try:
            self.module.validate_rotated_result(
                self.config,
                result_path=path,
                artifact_dir=ROOT / "tmp" / "does-not-exist",
                config_hash=self.config_hash,
                implementation_hash=self.implementation_hash,
                execution_policy="formal-a100",
            )
        finally:
            path.unlink()

    def test_w3_reference_is_scored_before_the_rotated_arm(self) -> None:
        self.assertEqual(self.module.ARMS, ("gptq_w3_g128_sym", ARM))

    def test_gate_rejects_a_mismatched_quantization_run(self) -> None:
        cases = {
            "wrong experiment": {"experiment_id": "qwen3-8b-hierarchical-w2-v1"},
            "wrong implementation": {"implementation_sha256": "c" * 64},
            "wrong config": {"config_sha256": "d" * 64},
            "unfinished": {"status": "partial_completed"},
            "wrong rotation": {"rotation": {"format": "offline-qwen3-r1-only"}},
            "wrong policy": {"execution": {"execution_policy": "same-device-quality"}},
        }
        for name, patch in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.check({**self.passing_result(), **patch})

    def test_gate_accepts_a_complete_runtime_bound_artifact_set(self) -> None:
        identity = runtime_identity(self.full_execution())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = []
            for layer in range(self.module.EXPECTED_LAYERS):
                layer_dir = root / f"layer-{layer:03d}"
                layer_dir.mkdir()
                payload = layer_dir / "payload.safetensors"
                payload.write_bytes(b"fixture")
                metadata = {
                    "status": "passed",
                    "arm": ARM,
                    "rotation_format": self.config["rotation"]["format"],
                    "runtime_identity": identity,
                    "linears": [
                        {"module": name} for name in self.module.QWEN3_LINEAR_MODULES
                    ],
                }
                metadata_path = layer_dir / "metadata.json"
                metadata_path.write_text(json.dumps(metadata))
                artifacts.append(
                    {
                        "metadata_sha256": self.module.sha256_file(metadata_path),
                        "payload_sha256": self.module.sha256_file(payload),
                    }
                )
            result = {
                **self.passing_result(),
                "execution": self.full_execution(),
                "coverage": {
                    "layer_count": self.module.EXPECTED_LAYERS,
                    "linear_count": self.module.EXPECTED_LINEARS,
                    "quantized_weight_count": self.module.EXPECTED_WEIGHTS,
                    "lm_head_quantized": False,
                    "no_fallback": True,
                },
                "artifacts": artifacts,
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            validated, records = self.module.validate_rotated_result(
                self.config,
                result_path=result_path,
                artifact_dir=root,
                config_hash=self.config_hash,
                implementation_hash=self.implementation_hash,
                execution_policy="formal-a100",
            )
            self.assertEqual(validated, result)
            self.assertEqual(len(records), self.module.EXPECTED_LAYERS)

    def test_recorded_evaluation_sources_exist(self) -> None:
        source = (ROOT / "scripts/run_qwen3_8b_hierarchical_w2_rotated_ppl.py").read_text()
        start = source.index('"evaluation_source_files_sha256"')
        block = source[start:source.index("},", start)]
        names = [line.strip().strip('",') for line in block.splitlines() if '.py"' in line]
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
