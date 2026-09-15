"""Offline inspection of a fixed candidate batch. Never launches/promotes trials."""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from .evaluation import sha256_file
from .deployment_artifacts import MANIFEST_SHA
from .qwen3 import QWEN3_LINEAR_MODULES, expected_qwen3_linear_shape
from .qwen3_8b import ARCHITECTURE


def timing(record):
    values=record['microseconds_per_call']
    if len(values)!=7 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('invalid timing samples')
    median=statistics.median(values)
    spread=(max(values)-min(values))/median
    stable=spread<=.10
    if record['stable']!=stable:
        raise ValueError('timing stability flag disagrees with samples')
    if not math.isclose(record['median_us'],median,rel_tol=1e-9):
        raise ValueError('timing median disagrees with samples')
    return median,stable


def summarize_batch(root: Path, config_path: Path):
    cfg=json.loads(config_path.read_text())
    batch=json.loads((root/'batch.json').read_text())
    if batch['config_sha256']!=sha256_file(config_path):raise ValueError('batch/config hash drift')
    entries=batch['trials']
    if len({e['id'] for e in entries})!=len(entries):raise ValueError('duplicate trial IDs')
    entries={e['id']:e for e in entries}
    expected={c['id']+'-'+mode for c in cfg['candidates'] for mode in cfg['modes']}
    if set(entries)-expected:raise ValueError('unexpected trial IDs')
    rows=[];provenance=None;trial_errors={};input_hashes={}
    for c in cfg['candidates']:
        for mode in cfg['modes']:
            trial=c['id']+'-'+mode;result=None;error=None
            try:
                entry=entries[trial];path=root/(trial+'.json')
                if sha256_file(path)!=entry['result_sha256']:raise ValueError('result hash drift')
                if sha256_file(root/(trial+'.log'))!=entry['log_sha256']:raise ValueError('log hash drift')
                result=json.loads(path.read_text())
                if entry['exit_code']!=0 or result['status'] not in ('completed_pending_review','completed_unstable'):
                    raise ValueError('failed/incomplete trial: '+result.get('status','unknown'))
                if entry['status']!=result['status']:raise ValueError('batch/result status drift')
                settings=result['settings']
                frozen={'kernel':c['kernel'],'groups_per_split':c['groups_per_split'],
                        'mode':mode,'dtype':cfg['dtype'],'warmup':20,'repeats':100,'rounds':7}
                if any(settings.get(k)!=v for k,v in frozen.items()):raise ValueError('settings drift')
                if result['stage']!='linear_m1' or result['layer']!=cfg['layer'] or result['manifest_sha256']!=MANIFEST_SHA:
                    raise ValueError('stage/layer/artifact drift')
                stamp={k:result[k] for k in ('source_sha256','environment_sha256','runner_sha256','payload')}
                if not all(stamp.values()):raise ValueError('missing provenance')
                if provenance is None:provenance=stamp
                if stamp!=provenance:raise ValueError('mixed source/environment/runner/payload')
                cells=result['cells']
                if len(cells)!=7 or {x['module'] for x in cells}!=set(QWEN3_LINEAR_MODULES):
                    raise ValueError('incomplete/duplicate Linear inventory')
            except (KeyError,ValueError,TypeError,OSError) as exc:
                error=str(exc);trial_errors[trial]=error
            cells={} if error else {x['module']:x for x in result['cells']}
            for module in QWEN3_LINEAR_MODULES:
                row={'trial':trial,'mode':mode,'kernel':c['kernel'],'groups_per_split':c['groups_per_split'],
                     'module':module,'shape':expected_qwen3_linear_shape(ARCHITECTURE,module),
                     'eligible':False,'reason':error,'dense_us':None,'packed_us':None,'v1_us':None,
                     'speedup_dense':None,'speedup_v1':None}
                if error is None:
                    try:
                        cell=cells[module]
                        if cell['shape']!=row['shape']:raise ValueError('shape drift')
                        digest=cell['input_sha256']
                        if not digest:raise ValueError('missing input hash')
                        if module in input_hashes and input_hashes[module]!=digest:raise ValueError('timing input drift')
                        input_hashes[module]=digest
                        checks=cell['checks']
                        if c['kernel'] in ('v4','v5'):
                            from .factored_reference import REFERENCE
                            if result.get('numerical_reference')!=REFERENCE or not all(x.get('reference')==REFERENCE for x in checks):
                                raise ValueError('factored structural reference missing')
                        if (len(checks)!=4 or not cell['correctness_passed'] or
                            not all(x.get('passed') is True for x in checks) or
                            not all(x.get('repeat_exact') is True for x in checks[:3]) or
                            (c['kernel'] not in ('v1','v3','v4','v5') and not all(x.get('v1_exact') is True for x in checks))):
                            raise ValueError('correctness gate failed/incomplete')
                        names=('dense','packed') if c['kernel']=='v1' else ('dense','packed','v1')
                        measured={n:timing(cell['timings'][n]) for n in names}
                        stable=all(t[1] for t in measured.values())
                        if cell['timing_stable']!=stable:raise ValueError('cell stability flag drift')
                        row.update({n+'_us':v[0] for n,v in measured.items()})
                        if not stable:raise ValueError('unstable timing')
                        row.update(eligible=True,reason=None,
                                   speedup_dense=measured['dense'][0]/measured['packed'][0],
                                   speedup_v1=measured['v1'][0]/measured['packed'][0] if 'v1' in measured else 1.0)
                    except (KeyError,ValueError,TypeError) as exc:row['reason']=str(exc)
                rows.append(row)
    winners=[]
    for mode in cfg['modes']:
        for module in QWEN3_LINEAR_MODULES:
            candidates=[r for r in rows if r['eligible'] and r['mode']==mode and r['module']==module]
            if candidates:
                winner=min(candidates,key=lambda r:r['packed_us'])
                winners.append({k:winner[k] for k in ('mode','module','trial','packed_us','speedup_dense')})
    return {'status':'summarized_pending_review','batch_status':batch['status'],
            'batch_sha256':sha256_file(root/'batch.json'),'config_sha256':sha256_file(config_path),
            'provenance':provenance,'trial_errors':trial_errors,'rows':rows,'winners':winners,
            'next_stage':'not_launched',
            'scope':'Per-shape fastest eligible measured configuration, not an automatic model-wide selection; modes kept separate.'}


def markdown(report):
    lines=['# M=1 candidate batch summary','',
           'Offline summary; no automatic acceptance or stage promotion. Missing/failed/unstable cells remain visible.',
           '', '| Trial | Mode | Module [O,K] | Dense us | Packed us | Dense/packed | v1/packed | Status |',
           '|---|---|---|---:|---:|---:|---:|---|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in report['rows']:
        state='eligible' if r['eligible'] else str(r['reason']).replace('|','/').replace('\n',' ')
        lines.append('| '+' | '.join([r['trial'],r['mode'],f"{r['module']} {r['shape']}",
                     fmt(r['dense_us']),fmt(r['packed_us']),fmt(r['speedup_dense']),fmt(r['speedup_v1']),state])+' |')
    lines+=['','## Fastest eligible configuration per shape and mode','',
            'Compare within the same source/environment/payload. These winners do not imply full-model speedup.','']
    for r in report['winners']:
        lines.append(f"- {r['mode']} / {r['module']}: {r['trial']}, {r['packed_us']:.3f} us, dense/packed {r['speedup_dense']:.3f}x")
    return '\n'.join(lines)+'\n'
