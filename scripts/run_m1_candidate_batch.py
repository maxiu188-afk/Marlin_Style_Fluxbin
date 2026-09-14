#!/usr/bin/env python3
"""Run the fixed bounded Linear candidate suite; never promote to block/model."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from fluxbin_style.evaluation import atomic_json, sha256_file

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/acceleration/m1_candidates_v1.json'


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'interrupted by signal {signum}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','environment','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIG)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    signal.signal(signal.SIGTERM,interrupted)
    if not 1<=len(cfg['candidates'])<=6 or len(cfg['modes'])!=2:
        raise ValueError('bounded batch contract changed')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'running','config_sha256':sha256_file(args.config),
            'runner_sha256':sha256_file(Path(__file__)),'trials':[],'next_stage':'not_launched'}
    try:
        for candidate in cfg['candidates']:
            for mode in cfg['modes']:
                name=candidate['id']+'-'+mode
                output=args.output_dir/(name+'.json');log=args.output_dir/(name+'.log')
                command=[sys.executable,str(ROOT/'scripts/run_m1_linear_benchmark.py'),
                         '--artifact-root',str(args.artifact_root),'--environment',str(args.environment),
                         '--output',str(output),'--layer',str(cfg['layer']),'--dtype',cfg['dtype'],
                         '--kernel',candidate['kernel'],'--groups-per-split',str(candidate['groups_per_split']),
                         '--mode',mode,'--warmup','20','--repeats','100','--rounds','7']
                with log.open('w') as stream:
                    proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=proc.wait(timeout=cfg['timeout_seconds_per_trial'])
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f'trial timed out: {name}')
                    finally:stop_process(proc)
                entry={'id':name,'exit_code':code,'log_sha256':sha256_file(log)}
                if output.exists():
                    result=json.loads(output.read_text())
                    entry.update(status=result['status'],result_sha256=sha256_file(output))
                report['trials'].append(entry);atomic_json(args.output_dir/'batch.json',report)
                if code!=0 or entry.get('status') not in ('completed_pending_review','completed_unstable'):
                    raise RuntimeError(f'candidate failed: {name}; inspect preserved result')
        report['status']='completed_pending_review'
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='failed',error=str(exc));raise
    finally:atomic_json(args.output_dir/'batch.json',report)


if __name__=='__main__':main()
