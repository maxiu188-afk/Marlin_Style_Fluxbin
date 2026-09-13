import unittest,json,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_qwen3_8b_distilled_step400_ppl as runner
sys.path.pop(0)
class DistilledPPLTests(unittest.TestCase):
    def test_protocol_and_gates_preserved_with_frozen_distillation_binding(self):
        c=json.loads(runner.CONFIG.read_text());runner.validate_config(c)
        parent=json.loads(runner.CONFIG.with_name('qwen3_8b_wikitext2_conditioned_hybrid_v1.json').read_text())
        for key in ('accepted_protocol','quality_gate','evaluation','model_preflight_files'):
            self.assertEqual(c[key],parent[key])
        c['accepted_distillation_result_sha256']='bad'
        with self.assertRaises(ValueError):runner.validate_config(c)
