import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from fluxbin_style import (
    capture_first_layer_inputs,
    capture_layer_hessians,
    invert_hessian,
)


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_qwen3_full_hessian_obq_s8.py"
    spec = importlib.util.spec_from_file_location("full_hessian_obq_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load full-model runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullHessianOBQRunnerTests(unittest.TestCase):
    def test_first_pass_and_resumed_payload_weights_are_identical(self) -> None:
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("transformers is unavailable")
        runner = load_runner()
        solver = {
            "max_iters": 1,
            "relative_tolerance": 1e-6,
            "convergence_patience": 1,
            "monotonicity_tolerance": 1e-6,
            "assignment_chunk_rows": 4,
            "denominator_epsilon": 1e-12,
        }
        experiment = {
            "model": {"revision": "tiny"},
            "algorithm": {
                "global_group_size": 4,
                "damp_percent": 0.01,
                "hybrid_s8": {"residual_columns_per_group": 2},
            },
            "global_solver": solver,
            "refinement_solver": solver,
        }
        model_config = Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
        )
        for arm in ("pure", "hybrid_s8"):
            with self.subTest(arm=arm):
                torch.manual_seed(61)
                model = Qwen3ForCausalLM(model_config).eval()
                tokens = torch.randint(0, model_config.vocab_size, (1, 4))
                first = capture_first_layer_inputs(
                    model,
                    tokens,
                    device=torch.device("cpu"),
                )
                layer = model.model.layers[0]
                capture = capture_layer_hessians(
                    layer,
                    first.inputs,
                    first.forward_kwargs,
                )
                payload = {}
                records = []
                for hessian_group, module_names in runner.HESSIAN_GROUP_MODULES.items():
                    hessian = capture.hessians[hessian_group]
                    inverse = invert_hessian(hessian, damp_percent=0.01)
                    for module_name in module_names:
                        records.append(
                            runner.add_quantized_module(
                                arm=arm,
                                module_name=module_name,
                                module=layer.get_submodule(module_name),
                                hessian=hessian,
                                inverse_hessian=inverse.inverse,
                                config=experiment,
                                payload=payload,
                            )
                        )
                expected = {
                    name: layer.get_submodule(name).weight.detach().clone()
                    for name in runner.QWEN3_LINEAR_MODULES
                }
                metadata = {
                    "schema_version": 2,
                    "status": "passed",
                    "arm": arm,
                    "layer_index": 0,
                    "config_sha256": "config",
                    "implementation_sha256": "implementation",
                    "model_revision": "tiny",
                    "linears": records,
                }
                with tempfile.TemporaryDirectory() as temporary:
                    layer_dir, _ = runner.write_completed_layer(
                        Path(temporary),
                        layer_index=0,
                        metadata=metadata,
                        payload=payload,
                    )
                    resumed = Qwen3ForCausalLM(model_config).model.layers[0]
                    runner.validate_and_apply_completed_layer(
                        layer_dir,
                        arm=arm,
                        layer_index=0,
                        layer=resumed,
                        config=experiment,
                        config_hash="config",
                        implementation_hash="implementation",
                        device=torch.device("cpu"),
                    )
                    for module_name, expected_weight in expected.items():
                        torch.testing.assert_close(
                            resumed.get_submodule(module_name).weight,
                            expected_weight,
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
