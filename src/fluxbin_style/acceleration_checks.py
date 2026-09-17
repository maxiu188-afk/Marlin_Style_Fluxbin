"""Numerical and repeated CUDA timing checks; no automatic stage promotion."""
from __future__ import annotations
import statistics
import torch


def numerical_gate(actual, expected):
    if actual.shape != expected.shape:
        return {'passed':False,'reason':'shape mismatch'}
    a,b=actual.detach().double(),expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {'passed':False,'reason':'nonfinite output'}
    rms=b.square().mean().sqrt().item()
    error=a-b
    maximum=error.abs().max().item()
    nrmse=error.square().mean().sqrt().item()/max(rms,1e-12)
    # Frozen v1 combined gate, including outputs close to zero.
    tol=.02 if expected.dtype==torch.bfloat16 else .002
    elementwise=bool((error.abs() <= tol*(rms+b.abs())+1e-6).all())
    return {'passed':elementwise and nrmse <= tol/4,
            'max_abs_error':maximum,'reference_rms':rms,'normalized_rmse':nrmse,
            'elementwise_tolerance_factor':tol,'normalized_rmse_limit':tol/4}


def stepwise_nrmse(actual, expected):
    """Per-step NRMSE along the step axis of a [1,steps,vocab] logits tensor.

    Report-only. Separates depth amplification (step 0) from autoregressive
    amplification (later steps); the frozen combined gate stays in
    numerical_gate. Returns None when the shape is not a step-indexed batch-1
    logits tensor.
    """
    if actual.shape!=expected.shape or actual.ndim!=3 or actual.shape[0]!=1:
        return None
    a,b=actual.detach().double(),expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return None
    rms=b.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    return [float(v) for v in ((a-b).square().mean(dim=-1).sqrt()/rms)[0]]


def paired_cuda_timing(functions, *, warmup=20, repeats=100, rounds=7):
    if warmup<1 or repeats<1 or rounds<3:
        raise ValueError('positive warmup/repeats and at least three rounds required')
    for fn in functions.values():
        for _ in range(warmup):fn()
    torch.cuda.synchronize()
    samples={name:[] for name in functions}
    names=list(functions)
    for round_index in range(rounds):
        # Alternate order to reduce systematic baseline/candidate ordering bias.
        for name in (names if round_index%2==0 else names[::-1]):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):functions[name]()
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/repeats)
    result={}
    for name,values in samples.items():
        median=statistics.median(values)
        spread=(max(values)-min(values))/median if median>0 else float('inf')
        result[name]={'microseconds_per_call':values,'median_us':median,
                      'relative_range':spread,'stable':spread<=.10}
    return result
