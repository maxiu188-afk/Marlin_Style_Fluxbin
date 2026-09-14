#!/usr/bin/env python3
"""Snapshot required build/runtime tools before and after first server setup.

Read-only except the requested new output directory. Never dumps environment
variables, pip URLs, credentials, or shell history. Run locally to test recorder;
only --require-cuda can mark an environment ready for CUDA experiments.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=30,cwd=ROOT)
        return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'exit_code':None,'error':type(exc).__name__}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('before','after','recreated'),required=True)
    p.add_argument('--image-reference',required=True,help='Non-secret template image tag/digest from provider')
    p.add_argument('--require-cuda',action='store_true')
    p.add_argument('--build-smoke',action='store_true',help='Explicitly compile and execute CUDA tests')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    versions={d.metadata['Name']:metadata.version(d.metadata['Name']) for d in metadata.distributions() if d.metadata['Name']}
    tools={k:command(v) for k,v in {
        'nvcc':['nvcc','--version'],'cxx':['c++','--version'],'ninja':['ninja','--version'],
        'git':['git','--version'],'tmux':['tmux','-V'],'nvidia_smi':['nvidia-smi'],
        'gpu_policy':['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory','--format=csv'],
        'os_release':['cat','/etc/os-release'],
        'system_packages':['dpkg-query','-W','-f=${binary:Package}=${Version}\n',
                           'build-essential','gcc','g++','ninja-build','git','tmux','ca-certificates'],
        'pip_check':[sys.executable,'-m','pip','check'],
        'git_revision':['git','rev-parse','HEAD'],'git_status':['git','status','--short'],
    }.items()}
    record={'schema_version':1,'phase':args.phase,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'image_reference':args.image_reference,'platform':platform.platform(),
            'machine':platform.machine(),'python':sys.version,'packages':versions,'tools':tools,
            'required_kernel':['torch with CUDA','CUDA toolkit with nvcc','compatible NVIDIA host driver','C++ compiler','ninja'],
            'required_model':['transformers','safetensors','pinned snapshot','accepted step400 payload'],
            'optional_research':['datasets (quality work only)','tmux (durable jobs)','git (source synchronization)'],
            'disk_free_bytes':shutil.disk_usage(args.output_dir).free,
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted((ROOT/'src/fluxbin_style').rglob('*')) if f.suffix in ('.py','.cu')},
            'build_flags':['-O3','--fmad=false','-lineinfo'],
            'candidate_extra_build_flags':{k:[f'-DROWS_PER_WARP={r}','--ptxas-options=-v']
                                           for k,r in [('v2_r1',1),('v2_r2',2),('v2',4)]} | {'v3':['--ptxas-options=-v']}}
    ready=False
    try:
        import torch
        record['torch']={'version':torch.__version__,'cuda_runtime':torch.version.cuda,
                         'cuda_available':torch.cuda.is_available(),
                         'cxx11_abi':getattr(torch._C,'_GLIBCXX_USE_CXX11_ABI',None),
                         'build_config':torch.__config__.show()}
        if torch.cuda.is_available():
            record['torch']['devices']=[{'name':torch.cuda.get_device_name(i),
                                        'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
        ready=(platform.system()=='Linux' and torch.cuda.is_available() and
               torch.cuda.get_device_capability(0)>=(8,0) and
               all(tools[k]['exit_code']==0 for k in ('nvcc','cxx','ninja','pip_check')))
    except ImportError:
        record['torch']={'import_error':True}
    if args.build_smoke:
        start=time.monotonic()
        # Run in a subprocess with a generous bounded timeout; no background jobs.
        try:
            proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_deployment.py','-v'],
                                cwd=ROOT,capture_output=True,text=True,timeout=600)
            record['build_smoke']={'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,
                                   'elapsed_seconds':time.monotonic()-start}
            ready=ready and proc.returncode==0
        except subprocess.TimeoutExpired:
            record['build_smoke']={'status':'timeout','elapsed_seconds':time.monotonic()-start};ready=False
    record['status']='ready_for_gpu_trial' if ready else 'not_gpu_ready'
    (args.output_dir/'environment.json').write_text(json.dumps(record,indent=2)+'\n')
    # Name/version only: avoid pip freeze's editable URLs, tokens and local paths.
    (args.output_dir/'packages.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(versions.items(),key=lambda t:t[0].lower())))
    print(record['status'])
    if args.require_cuda and not ready:raise SystemExit(1)


if __name__=='__main__':main()
