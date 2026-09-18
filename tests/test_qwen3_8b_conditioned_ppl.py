import json,sys,copy,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_qwen3_8b_conditioned_hybrid_ppl as r
sys.path.pop(0)
class ConditionedPPLTests(unittest.TestCase):
    def test_protocol_unchanged_and_no_pure(self):
        c=json.loads(r.CONFIG.read_text());r.validate_config(c)
        old=json.loads(r.CONFIG.with_name('qwen3_8b_wikitext2_full_hessian_obq_s8_v1.json').read_text())
        self.assertEqual(c['accepted_protocol'],old['accepted_protocol'])
        self.assertEqual(c['quality_gate'],old['quality_gate'])
        self.assertEqual(c['evaluation']['arms'],['bf16','hybrid_s8'])
        c['accepted_protocol']['full_block_count']=1
        with self.assertRaises(ValueError):r.validate_config(c)
    def test_quality_not_conflated_with_execution(self):
        a={n:dict(perplexity=p,mean_nll=math.log(p),metrics_valid=True,scored_transition_count=298862) for n,p in [('bf16',10.),('hybrid_s8',12.)]}
        valid,g=r.assess_quality(a,298862,dict(max_relative_gap=.05,deployment_block_relative_gap=.1))
        self.assertTrue(valid);self.assertFalse(g['hybrid_s8']['quality_gate_passed']);self.assertTrue(g['hybrid_s8']['deployment_explicitly_blocked'])
        a['hybrid_s8']['scored_transition_count']=1
        self.assertFalse(r.assess_quality(a,298862,dict(max_relative_gap=.05,deployment_block_relative_gap=.1))[0])
