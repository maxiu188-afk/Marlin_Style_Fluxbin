"""Fail-closed progression checks for explicit full-model trials."""
from .deployment import KERNELS
from .deployment_artifacts import MANIFEST_SHA


def validate_block_gate(block, *, source_sha256, environment_sha256):
    if (block.get('status')!='completed_pending_review'
        or block.get('stage')!='single_block_empty_cache_m1'
        or block.get('layer')!=0 or block.get('manifest_sha256')!=MANIFEST_SHA
        or block.get('kernel_coverage')!=7
        or block.get('source_sha256')!=source_sha256
        or block.get('environment_sha256')!=environment_sha256):
        raise ValueError('matching completed layer-0 block gate required')
    if block.get('kernel') not in KERNELS or not 1<=block.get('groups_per_split',0)<=1024:
        raise ValueError('unknown kernel/split contract')
    checks=block.get('checks',[])
    if len(checks)!=3 or not all(c.get('passed') and c.get('repeat_exact') for c in checks):
        raise ValueError('block numerical/repeat gates not passed')
    timings=block.get('timings',{})
    if set(timings)!={'original_bf16','decoded_step400_bf16','packed_step400'}:
        raise ValueError('incomplete block timing')
    import math,statistics
    for t in timings.values():
        values=t.get('microseconds_per_call',[])
        if len(values)!=7 or not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('invalid block timing samples')
        median=statistics.median(values)
        if not t.get('stable') or (max(values)-min(values))/median>.10:
            raise ValueError('unstable block timing')
    return block['kernel'],block['groups_per_split']
