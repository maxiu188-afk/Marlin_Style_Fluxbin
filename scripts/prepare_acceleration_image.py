#!/usr/bin/env python3
"""Generate an image build bundle from two verified first-server snapshots.

Does NOT build, publish, connect to a server or provision compute. Template
software is retained; only observed package additions/changes are installed.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def prepare(before,after,output):
    if after.get('status')!='ready_for_gpu_trial' or after.get('build_smoke',{}).get('exit_code')!=0:
        raise ValueError('after snapshot must include passing CUDA build/smoke')
    base=after['image_reference']
    if before['image_reference']!=base or not re.fullmatch(r'[a-zA-Z0-9._:/-]+@sha256:[a-f0-9]{64}',base):
        raise ValueError('both records must identify the same immutable base image digest')
    if not after['machine'] in ('x86_64','AMD64') or 'Linux' not in after['platform']:
        raise ValueError('first bundle targets Linux x86_64')
    if before['python']!=after['python']:
        raise ValueError('Python changed: select a matching base before preparing an image')
    def normalized(record):
        return {re.sub(r'[-_.]+','-',k).lower():v for k,v in record['packages'].items()}
    old,new=normalized(before),normalized(after)
    for key in ('torch','triton','numpy'):
        if old.get(key)!=new.get(key):raise ValueError(f'base {key} changed; resolve base image first')
    addons={k:v for k,v in new.items() if old.get(k)!=v and k!='marlin-style-fluxbin'}
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*',k) or
           not re.fullmatch(r'[A-Za-z0-9.+!_-]+',v) for k,v in addons.items()):
        raise ValueError('unexpected package name/version; inspect snapshot')
    package_record=after['tools']['system_packages']
    if package_record['exit_code']!=0:
        raise ValueError('system package inventory incomplete; record required toolchain first')
    apt=package_record['stdout'].splitlines()
    if not apt or any(not re.fullmatch(r'[a-z0-9+.:_-]+=[A-Za-z0-9+.:~_-]+',line) for line in apt):
        raise ValueError('unexpected Debian package inventory')
    output.mkdir(parents=True,exist_ok=False)
    (output/'addons.lock').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(addons.items())))
    dockerfile=f'''# Generated from passing first-server records. Review before building.
FROM {base}
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends {' '.join(apt)} && rm -rf /var/lib/apt/lists/*
COPY addons.lock /opt/fluxbin/addons.lock
RUN python -m pip install --no-cache-dir --no-deps -r /opt/fluxbin/addons.lock && python -m pip check
# Model, credentials, experiment code and mutable caches stay outside this image.
WORKDIR /workspace
CMD ["sleep", "infinity"]
'''
    (output/'Dockerfile').write_text(dockerfile)
    for name,data in (('before',before),('after',after)):
        (output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    (output/'README.md').write_text('''Generated build context only; no image has been built or published.

Build for linux/amd64 on a builder with enough disk. Pin the produced image by
registry digest. Recreate a trial Pod with the original network volume, re-link
the repository with python -m pip install --no-deps -e ., configure cache paths
with infra/runpod/entrypoint.sh, and repeat --require-cuda --build-smoke using
the new image digest. Only then accept the image. Record cold setup and extension
compile times separately. A warm extension cache is not a compiler correctness gate.

The environment ID must change with Python/torch/CUDA/compiler ABI changes.
The host NVIDIA driver belongs to the provider, not the container image.
Do not auto-pull repository code or run experiments from the image entrypoint.
''')
    return {'base':base,'addon_count':len(addons),'status':'prepared_not_built'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    print(json.dumps(prepare(json.loads(args.before.read_text()),json.loads(args.after.read_text()),args.output_dir)))


if __name__=='__main__':main()
