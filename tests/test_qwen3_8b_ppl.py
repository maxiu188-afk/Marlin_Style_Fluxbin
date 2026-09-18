import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_qwen3_8b_full_hessian_obq_s8_ppl as runner
sys.path.pop(0)


class Qwen3EightBPPLTests(unittest.TestCase):
    def test_frozen_protocol_rejects_resampling_and_wrong_model(self):
        original = json.loads(runner.CONFIG.read_text())
        runner.validate_config(original)
        for field, name, value in [('model', 'repo_id', 'Qwen/Qwen3-32B'),
                                    ('accepted_protocol', 'full_block_count', 1),
                                    ('evaluation', 'use_cache', True)]:
            config = copy.deepcopy(original)
            config[field][name] = value
            with self.assertRaises(ValueError):
                runner.validate_config(config)

    def test_quality_boundaries_and_invalid_counts(self):
        arms = {name: {'perplexity': ppl, 'mean_nll': math.log(ppl),
                      'metrics_valid': True, 'scored_transition_count': 298862}
                for name, ppl in [('bf16', 10.), ('pure', 11.01), ('hybrid_s8', 10.5)]}
        gate = {'max_relative_gap': .05, 'deployment_block_relative_gap': .1}
        valid, result = runner.assess_quality(arms, 298862, gate)
        self.assertTrue(valid)
        self.assertTrue(result['hybrid_s8']['quality_gate_passed'])
        self.assertTrue(result['pure']['deployment_explicitly_blocked'])
        arms['pure']['scored_transition_count'] -= 1
        valid, result = runner.assess_quality(arms, 298862, gate)
        self.assertFalse(valid)
        self.assertFalse(result['hybrid_s8']['quality_gate_passed'])
        arms['pure']['scored_transition_count'] += 1
        arms['bf16']['perplexity'] = float('nan')
        self.assertFalse(runner.assess_quality(arms, 298862, gate)[0])

    def test_scorer_matches_manual_next_token_nll_without_block_boundary(self):
        # Fixed logits: compare chunked inherited scorer with a direct FP32 CE oracle.
        logits = torch.tensor([[[2., 1., -1.], [0., 3., 1.], [1., -1., 2.]]])
        class Model:
            def __call__(self, input_ids, use_cache):
                return type('Output', (), {'logits': logits})()
        blocks = [[0, 1, 2], [2, 0, 1]]
        expected = sum(float(torch.nn.functional.cross_entropy(
            logits[0, :-1], torch.tensor(b[1:]), reduction='sum')) for b in blocks)
        with patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.max_memory_allocated', return_value=0):
            result = runner.score_model(Model(), blocks, arm='bf16', device=torch.device('cpu'), logit_chunk_tokens=1)
        self.assertEqual(result['scored_transition_count'], 4)
        self.assertAlmostEqual(result['total_nll'], expected, places=6)
        self.assertAlmostEqual(result['perplexity'], math.exp(expected/4), places=5)


if __name__ == '__main__':
    unittest.main()
