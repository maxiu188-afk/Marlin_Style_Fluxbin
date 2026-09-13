import importlib.util
import json
import tempfile
import unittest
import torch
from pathlib import Path
from unittest import mock

import test_full_hessian_obq_runner as historical
from fluxbin_style import sha256_file
from fluxbin_style.qwen3_8b import TARGETS, REVISION
from fluxbin_style.qwen3_8b_full import validate_suite, relative_file

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("full8b", ROOT / "scripts/run_qwen3_8b_full_hessian_obq_s8.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Full8BTests(unittest.TestCase):
    def test_conditioned_full_route_matches_probe_and_packed_replay(self):
        r=runner();torch.manual_seed(31)
        module=torch.nn.Linear(16,9,bias=False)
        w=module.weight.detach().clone();x=torch.randn(50,16);h=x.T@x/25
        inv=torch.linalg.inv(h+.01*torch.eye(16))
        solver={'max_iters':3}
        c={'algorithm':{'global_group_size':8,'hybrid_s8':{'residual_columns_per_group':2,
            'compensation':'conditioned_fixed_indices_v1','fixed_indices_policy':'legacy_fit_on_same_target_and_hessian'}},
            'global_solver':solver,'refinement_solver':solver}
        cfg=r.solver_config(solver)
        legacy=r.quantize_hybrid_two_base_obq(w,inv,group_size=8,columns_per_group=2,global_config=cfg,refinement_config=cfg)
        fixed=legacy.decomposition.selected_indices
        expected=r.quantize_hybrid_conditioned_v1(w,inv,fixed_indices=fixed,group_size=8,columns_per_group=2,global_config=cfg,refinement_config=cfg)
        payload={}
        rec=r.add_quantized_module(arm='hybrid_s8',module_name='mlp.down_proj',module=module,hessian=h,inverse_hessian=inv,config=c,payload=payload)
        self.assertEqual(rec['legacy_selected_indices_sha256'],rec['conditioned_selected_indices_sha256'])
        torch.testing.assert_close(payload['mlp.down_proj.refinement_indices'].long(),fixed,rtol=0,atol=0)
        torch.testing.assert_close(module.weight,expected.decomposition.reconstruct().bfloat16().float(),rtol=0,atol=0)
        replay=r.materialize_module(payload,'mlp.down_proj',arm='hybrid_s8',config=c,device='cpu')
        torch.testing.assert_close(module.weight,replay.float(),rtol=0,atol=0)

    def test_conditioned_gate_rejects_pure_and_missing_probe(self):
        from types import SimpleNamespace
        from fluxbin_style.qwen3_8b_full import validate_full_contract
        c=json.loads((ROOT/'configs/experiments/qwen3_8b_full_hybrid_conditioned_v1.json').read_text())
        with self.assertRaisesRegex(ValueError,'hybrid only'):
            validate_full_contract(c,SimpleNamespace(arm='pure'))
        with self.assertRaisesRegex(ValueError,'probe is required'):
            validate_full_contract(c,SimpleNamespace(arm='hybrid_s8'))

    def test_same_resume_oracle_on_new_runner(self):
        # Reuse the established tiny-model oracle, with the 8B runner under test.
        with mock.patch.object(historical, "load_runner", runner):
            historical.FullHessianOBQRunnerTests().test_first_pass_and_resumed_payload_weights_are_identical()

    def test_36_layer_bounded_execution(self):
        r = runner()
        self.assertEqual(r.execution_layer_count(36, None), 36)
        self.assertEqual(r.execution_layer_count(36, 0), 1)
        self.assertEqual(r.execution_layer_count(36, 35), 36)
        with self.assertRaises(ValueError):
            r.execution_layer_count(36, 36)

    def fixture(self, root, fail=False):
        entries = []
        for index, (module, shape) in enumerate(TARGETS.items()):
            name = f"model.layers.0.{module}.weight"
            c = json.loads((ROOT / "configs/experiments/qwen3_8b_single_linear_hessian_obq_s8_v1.json").read_text())
            c["model"].update(target_tensor=name, target_shape=shape, target_tensor_sha256="a"*64)
            c["preflight"] = {"status":"passed", "revision":REVISION,
                "files":{n:"b"*64 for n in ("config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json")},
                "target_tensor_sha256":{name:"a"*64}}
            cp, pp, rp, ap = (root/f"{index}.{ext}" for ext in ("config","payload","result","acceptance"))
            cp.write_text(json.dumps(c)); pp.write_bytes(b"fixture")
            entry = {"target":name,"config":cp.name,"payload":pp.name,"result":rp.name,"acceptance":ap.name}
            entry.update(config_sha256=sha256_file(cp),payload_sha256=sha256_file(pp))
            rp.write_text(json.dumps({"status":"completed_pending_review","model":{"target_tensor":name,"revision":REVISION},
                "decision":{"linear_gate_passed":True},"config_sha256":entry["config_sha256"],
                "payload":{"sha256":entry["payload_sha256"]},"calibration":{"artifact_manifest_sha256":"calibration"},
                "reconstruction":{"pure_two_base_obq":{"squared_error":2.,"calibration_total_output_squared_error":2.},
                    "hessian_salient_hybrid_s8_obq":{"squared_error":3. if fail and index==4 else 1.,"calibration_total_output_squared_error":1.},
                    "hybrid_maximum_delta_outside_selected_columns":0.}}))
            entry["result_sha256"] = sha256_file(rp)
            ap.write_text(json.dumps({"status":"passed","target":name,**{k:entry[k] for k in ("config_sha256","payload_sha256","result_sha256")}}))
            entry["acceptance_sha256"] = sha256_file(ap)
            entries.append(entry)
        return {"status":"passed","entries":entries,"calibration_manifest_sha256":"calibration"}

    def test_suite_requires_five_passed_immutable_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = self.fixture(root)
            validate_suite(suite, root)
            with self.assertRaises(ValueError):
                validate_suite({**suite,"entries":suite["entries"][:-1]},root)
            (root/suite["entries"][0]["payload"]).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                validate_suite(suite,root)

    def test_self_reported_pass_cannot_override_bad_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                validate_suite(self.fixture(root,fail=True),root)

    def test_suite_paths_cannot_escape_root(self):
        for name in ("../payload", "/payload"):
            with self.assertRaises(ValueError):
                relative_file(ROOT,name)


if __name__ == "__main__":
    unittest.main()
