"""Synthetic reports exercise offline validation, never stand in for GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from fluxbin_style.candidate_summary import summarize_batch,markdown
from fluxbin_style.deployment_artifacts import MANIFEST_SHA
from fluxbin_style.evaluation import sha256_file
from fluxbin_style.qwen3 import QWEN3_LINEAR_MODULES,expected_qwen3_linear_shape
from fluxbin_style.qwen3_8b import ARCHITECTURE

CONFIG=Path(__file__).resolve().parents[1]/'configs/acceleration/m1_candidates_v1.json'


def fixture(root):
    batch={'status':'running','config_sha256':sha256_file(CONFIG),'trials':[]}
    for cid,kernel,us in [('baseline','v1',20),('rows2','v2_r2',12)]:
        name=cid+'-graph'
        check={'passed':True,'repeat_exact':True,'v1_exact':True}
        timings={n:{'microseconds_per_call':[v]*7,'median_us':v,'stable':True}
                 for n,v in [('dense',10),('packed',us),('v1',20)]}
        report={'status':'completed_pending_review','stage':'linear_m1','layer':0,
                'manifest_sha256':MANIFEST_SHA,'source_sha256':{'synthetic':'only'},
                'environment_sha256':'synthetic','runner_sha256':'synthetic','payload':{'sha256':'synthetic'},
                'settings':{'kernel':kernel,'groups_per_split':8,'mode':'graph','dtype':'bf16',
                            'warmup':20,'repeats':100,'rounds':7},
                'cells':[{'module':m,'shape':expected_qwen3_linear_shape(ARCHITECTURE,m),
                          'input_sha256':'synthetic-input-'+m,
                          'checks':[copy.deepcopy(check) for _ in range(4)],'correctness_passed':True,
                          'timing_stable':True,'timings':copy.deepcopy(timings)} for m in QWEN3_LINEAR_MODULES]}
        (root/(name+'.json')).write_text(json.dumps(report))
        (root/(name+'.log')).write_text('synthetic fixture\n')
        batch['trials'].append({'id':name,'exit_code':0,'status':report['status'],
                               'result_sha256':sha256_file(root/(name+'.json')),
                               'log_sha256':sha256_file(root/(name+'.log'))})
    (root/'batch.json').write_text(json.dumps(batch))


def mutate(root,fn):
    path=root/'rows2-graph.json';value=json.loads(path.read_text());fn(value)
    path.write_text(json.dumps(value))
    batch=json.loads((root/'batch.json').read_text())
    batch['trials'][1]['result_sha256']=sha256_file(path)
    (root/'batch.json').write_text(json.dumps(batch))


class CandidateSummaryTests(unittest.TestCase):
    def test_partial_batch_keeps_all_cells_and_separates_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root);report=summarize_batch(root,CONFIG)
            self.assertEqual(len(report['rows']),84)
            self.assertEqual(sum(r['eligible'] for r in report['rows']),14)
            self.assertEqual(len(report['winners']),7)
            self.assertTrue(all(r['trial']=='rows2-graph' for r in report['winners']))
            winner=report['winners'][0]
            self.assertAlmostEqual(winner['speedup_dense'],10/12)
            self.assertIn('baseline-eager',markdown(report))
            self.assertEqual(report['next_stage'],'not_launched')

    def test_corrupt_mixed_failed_and_invalid_timings_are_excluded(self):
        mutations=[lambda r:r.update(environment_sha256='other'),
                   lambda r:r.update(status='failed'),
                   lambda r:r['cells'][0].update(input_sha256='different'),
                   lambda r:r['cells'][0]['timings']['packed'].update(microseconds_per_call=[float('nan')]*7),
                   lambda r:r['cells'][0]['timings']['packed'].update(median_us=.01),
                   lambda r:r['cells'][0]['checks'][0].update(v1_exact=False)]
        for change in mutations:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);fixture(root);mutate(root,change)
                report=summarize_batch(root,CONFIG)
                row=next(r for r in report['rows'] if r['trial']=='rows2-graph')
                self.assertFalse(row['eligible']);self.assertIsNone(row['speedup_dense'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            with (root/'rows2-graph.json').open('a') as f:f.write(' ')
            report=summarize_batch(root,CONFIG)
            self.assertIn('hash drift',report['trial_errors']['rows2-graph'])

    def test_unstable_samples_do_not_receive_speedup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            def unstable(r):
                r['cells'][0]['timings']['packed'].update(microseconds_per_call=[12]*6+[30],stable=False)
                r['cells'][0]['timing_stable']=False
            mutate(root,unstable)
            row=next(r for r in summarize_batch(root,CONFIG)['rows'] if r['trial']=='rows2-graph')
            self.assertEqual(row['reason'],'unstable timing')
            self.assertEqual(row['packed_us'],12)
            self.assertIsNone(row['speedup_dense'])

if __name__=='__main__':unittest.main()
