import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from fluxbin_style import (
    build_qwen3_linear_inventory, expected_qwen3_linear_shape,
    qwen3_linear_weight_names, tensor_sha256, sha256_file,
    quantize_pure_two_base_obq, quantize_hybrid_two_base_obq,
    TwoBaseRankOneOptimizationConfig, invert_hessian,
)
from fluxbin_style.qwen3_8b import (
    ARCHITECTURE, REVISION, TARGETS, linear_gate, validate_architecture,
    validate_linear_config, snapshot_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location(
        "linear8b", ROOT / "scripts/run_qwen3_8b_single_linear_hessian_obq_s8.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen3EightBTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/experiments/qwen3_8b_single_linear_hessian_obq_s8_v1.json").read_text())

    def resolved(self):
        c = copy.deepcopy(self.config)
        c["model"]["target_tensor_sha256"] = "a" * 64
        c["preflight"] = {
            "status": "passed", "revision": REVISION,
            "files": {n: "b" * 64 for n in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")},
            "target_tensor_sha256": {c["model"]["target_tensor"]: "a" * 64},
        }
        return c

    def test_expected_8b_inventory_and_shapes(self):
        records = []
        for name in qwen3_linear_weight_names(36):
            module = name.split(".", 3)[3].removesuffix(".weight")
            records.append({"name": name, "shape": expected_qwen3_linear_shape(ARCHITECTURE, module)})
        inventory = build_qwen3_linear_inventory(ARCHITECTURE, records)
        self.assertEqual(inventory["included_tensor_count"], 252)
        self.assertEqual(inventory["included_parameter_count"], 6945767424)
        for module, shape in TARGETS.items():
            self.assertEqual(expected_qwen3_linear_shape(ARCHITECTURE, module), shape)

    def test_32b_architecture_and_unresolved_config_rejected(self):
        with self.assertRaises(ValueError):
            validate_architecture({**ARCHITECTURE, "num_hidden_layers": 64})
        with self.assertRaisesRegex(ValueError, "preflight"):
            validate_linear_config(self.config)

    def test_no_silent_algorithm_or_layer_change(self):
        validate_linear_config(self.resolved())
        for key in ("algorithm", "global_solver", "calibration"):
            c = self.resolved()
            c[key] = {}
            with self.assertRaises(ValueError):
                validate_linear_config(c)
        c = self.resolved()
        c["model"]["target_tensor"] = "model.layers.1.self_attn.o_proj.weight"
        with self.assertRaises(ValueError):
            validate_linear_config(c)

    def test_preflight_path_escape_rejected(self):
        c = self.resolved()
        c["preflight"]["files"]["../other"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_linear_config(c)

    def test_snapshot_rejects_wrong_revision_before_io(self):
        with self.assertRaises(ValueError):
            snapshot_preflight(Path("wrong-revision"))

    def test_gate_requires_both_errors_to_improve_and_finite_support(self):
        p = {"squared_error": 2., "calibration_total_output_squared_error": 4.}
        h = {"squared_error": 1., "calibration_total_output_squared_error": 3.}
        self.assertTrue(linear_gate(p, h, 0)["linear_gate_passed"])
        for candidate, outside in ((p, 0), (h, 1e-9), ({**h, "squared_error": float("nan")}, 0), ({**h, "calibration_total_output_squared_error": 5.}, 0)):
            self.assertFalse(linear_gate(p, candidate, outside)["linear_gate_passed"])
        self.assertFalse(linear_gate(p, h, 0)["full_model_auto_launch"])

    def test_calibration_seed_tokenizer_and_file_integrity(self):
        r = runner()
        config = self.resolved()
        canonical = json.loads((ROOT / "configs/calibration/qwen3_8b_c4_256x2048_v1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokens = torch.zeros(256, 2048, dtype=torch.int32)
            path, manifest_path = root / "tokens.safetensors", root / "manifest.json"
            save_file({"token_ids": tokens}, path)
            manifest = {"status": "passed", "artifact_id": canonical["artifact_id"], "config": canonical,
                        "tokenizer": {"files": {n: "b" * 64 for n in ("tokenizer.json", "tokenizer_config.json")}},
                        "tokens": {"file_sha256": sha256_file(path), "tensor_sha256": tensor_sha256(tokens)}}
            manifest_path.write_text(json.dumps(manifest))
            actual, _ = r.load_calibration(config, manifest_path, path)
            self.assertEqual(actual.dtype, torch.int64)
            for mutate in (lambda m: m["config"].update(seed=0),
                           lambda m: m["tokenizer"]["files"].update({"tokenizer.json": "wrong"}),
                           lambda m: m["tokens"].update(file_sha256="wrong")):
                bad = copy.deepcopy(manifest)
                mutate(bad)
                manifest_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    r.load_calibration(config, manifest_path, path)

    def test_tiny_qwen_capture_matches_direct_activation_hessian(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(13)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4, max_position_embeddings=32)).eval()
        tokens = torch.randint(0, 64, (2, 8))
        for module_name in ("self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
            module = model.get_submodule("model.layers.0." + module_name)
            inputs = []
            handle = module.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().reshape(-1, module.in_features)))
            with torch.inference_mode():
                for row in tokens:
                    model(input_ids=row[None], use_cache=False)
            handle.remove()
            x = torch.cat(inputs)
            expected = 2 * x.T @ x / len(x)
            actual, count, _ = runner().capture_target_hessian(model, tokens, target_module=module, device=torch.device("cpu"))
            torch.testing.assert_close(actual, expected)
            self.assertEqual(count, 16)

    def test_real_solver_small_two_group_contract(self):
        torch.manual_seed(17)
        weight = torch.randn(8, 256)
        hessian = torch.eye(256)
        inverse = invert_hessian(hessian, damp_percent=.01).inverse
        solver = TwoBaseRankOneOptimizationConfig(max_iters=50)
        pure = quantize_pure_two_base_obq(weight, inverse, group_size=128, config=solver)
        hybrid = quantize_hybrid_two_base_obq(weight, inverse, group_size=128, columns_per_group=8, global_config=solver, refinement_config=solver)
        r = runner()
        pm = r.reconstruction_metrics(weight, pure.decomposition.reconstruct(), hessian)
        hm = r.reconstruction_metrics(weight, hybrid.decomposition.reconstruct(), hessian)
        self.assertTrue(linear_gate(pm, hm, 0)["linear_gate_passed"])


if __name__ == "__main__":
    unittest.main()
