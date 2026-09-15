#!/usr/bin/env python3
"""Create/reuse a Linux venv on a persistent mount, guarded by base-runtime identity.

Run with the template's base Python, not an activated venv. Never copies an
existing venv. OS tools/Python/CUDA remain dependencies of the container image.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def command(args, *, timeout=30, **kwargs):
    return subprocess.check_output(args, text=True, timeout=timeout, **kwargs).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--persist-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--image-reference', default='runpod-default-unresolved')
    p.add_argument('--resume-incomplete', action='store_true',
                   help='Validate an interrupted installation without reinstalling; all checks must pass')
    args = p.parse_args()
    if platform.system() != 'Linux' or sys.prefix != sys.base_prefix:
        raise RuntimeError('Run on Linux with the template base Python, outside any venv')
    if not args.persist_root.is_dir() or not args.persist_root.is_mount():
        raise ValueError('persist-root must be an existing mount point; verify it is the persistent volume')
    for tool in ('nvcc', 'c++', 'ninja'):
        if not shutil.which(tool):
            raise RuntimeError(f'{tool} missing in base PATH; restore OS build tools first')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required for the cache namespace')
    lock = ROOT/'infra/runpod/requirements-linear-a100-v1.lock'
    identity = {
        'schema':1, 'image_reference':args.image_reference,
        'system':platform.system(), 'machine':platform.machine(), 'libc':platform.libc_ver(),
        'os_release':Path('/etc/os-release').read_text(), 'python':sys.version,
        'executable':str(Path(sys.executable).resolve()), 'repo':str(ROOT),
        'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
        'cuda_arch':'.'.join(map(str,torch.cuda.get_device_capability())),
        'nvcc':command(['nvcc','--version']), 'cxx':command(['c++','--version']),
        'ninja':command(['ninja','--version']),
        'base_packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()
                               if d.metadata['Name'] and d.metadata['Name'].lower() != 'marlin-style-fluxbin'),
        'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
        'project_metadata_sha256':hashlib.sha256((ROOT/'pyproject.toml').read_bytes()).hexdigest(),
    }
    key = fingerprint(identity)
    root = args.persist_root.resolve()
    bundle = root/'environments'/key
    envdir = bundle/'venv'
    marker = bundle/'ready.json'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    # Exclusive creation prevents two launchers mutating the same environment.
    exists = bundle.exists()
    reused = marker.exists()
    if exists:
        if (not reused and not args.resume_incomplete) or (reused and json.loads(marker.read_text())['fingerprint'] != key):
            raise RuntimeError(f'Incomplete or mismatched environment: {bundle}; inspect before retrying')
    else:
        bundle.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PIP_CACHE_DIR=str(root/'cache/pip'))
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(envdir)], check=True)
        py = str(envdir/'bin/python')
        subprocess.run([py,'-m','pip','install','-r',str(lock)], env=env, check=True)
        subprocess.run([py,'-m','pip','install','--no-deps','-e',str(ROOT)], env=env, check=True)
    py = str(envdir/'bin/python')
    subprocess.run([py,'-m','pip','check'], check=True)
    command([py,'-c',
             'import importlib.metadata as m,sys; from pathlib import Path; '
             'pairs=[s.strip().split("==") for s in Path(sys.argv[1]).read_text().splitlines() '
             'if s.strip() and not s.lstrip().startswith("#")]; '
             'assert all(m.version(n)==v for n,v in pairs), "installed lock versions drifted"', str(lock)])
    import_start = time.monotonic()
    command([py,'-c','import torch,transformers,safetensors,fluxbin_style; '
             'assert torch.cuda.is_available(); print(torch.__version__)'], timeout=120)
    import_seconds = time.monotonic()-import_start
    if not reused:
        marker.write_text(json.dumps({'fingerprint':key,'identity':identity}, indent=2)+'\n')
    cuda_root = Path(shutil.which('nvcc')).resolve().parents[1]
    exports = {'CUDA_HOME':str(cuda_root), 'TORCH_CUDA_ARCH_LIST':identity['cuda_arch'],
               'TORCH_EXTENSIONS_DIR':str(root/'cache/torch-extensions'/key),
               'PIP_CACHE_DIR':str(root/'cache/pip'), 'MAX_JOBS':'2',
               'HF_HOME':str(root/'cache/huggingface')}
    lines = ['# Generated runtime for this fingerprint and fixed mount path.',
             'source '+shlex.quote(str(envdir/'bin/activate'))]
    lines += [f'export {k}={shlex.quote(v)}' for k,v in exports.items()]
    lines += ['export PATH="$CUDA_HOME/bin:$PATH"']
    (args.output_dir/'runtime.sh').write_text('\n'.join(lines)+'\n')
    report = {'status':'runtime_imports_passed_gpu_kernel_validation_still_required',
              'fingerprint':key, 'reused':reused, 'identity':identity,
              'resumed_incomplete':exists and not reused,
              'venv':str(envdir), 'elapsed_seconds':time.monotonic()-start,
              'import_seconds':import_seconds,
              'limits':'Fixed path/base-runtime reuse only; no binary/content integrity guarantee or GPU performance claim'}
    (args.output_dir/'runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('fingerprint','reused','elapsed_seconds','import_seconds')}))


if __name__ == '__main__':
    main()
