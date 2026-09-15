"""Independent high-precision structural reference for v4, never the timed baseline."""
import torch
from .acceleration_checks import numerical_gate

REFERENCE = 'hybrid-structural-fp64-v1'


@torch.no_grad()
def structural_reference(x, layout):
    """FP32 payload coefficients promoted to FP64 before reconstruction/dot.

    No BF16 weight rounding. This is high precision, not exact real arithmetic.
    Only one group's dense weights exist at a time; no full-model dense copy.
    """
    g, o, _ = layout['codes'].shape
    if tuple(x.shape) != (1, g*128):raise ValueError('expected [1,G*128]')
    result = torch.zeros(o, device=x.device, dtype=torch.float64)
    k = torch.arange(128, device=x.device)
    j = torch.arange(8, device=x.device)
    for group in range(g):
        q = layout['codes'][group][:,k//4].long() >> (2*(k%4))
        signs = torch.stack(((q&1)*2-1, ((q>>1)&1)*2-1), dim=-1).double()
        rows = layout['rows'][group].double()
        cols = layout['columns'][group].double()
        w = (signs * rows[:,None,:] * cols[None,:,:]).sum(-1)
        sq = layout['sparse_codes'][group][:,j//4].long() >> (2*(j%4))
        ss = torch.stack(((sq&1)*2-1, ((sq>>1)&1)*2-1), dim=-1).double()
        delta = (ss * layout['sparse_rows'][group].double()[:,None,:] *
                 layout['sparse_columns'][group].double()[None,:,:]).sum(-1)
        w[:,layout['indices'][group].long()] += delta
        result += w @ x[0,group*128:(group+1)*128].double()
    return result[None,:]


def structural_gate(actual, expected):
    # Keep the existing output-dtype tolerances, but compare directly to FP64
    # structural output rather than BF16-rounded weights or rounded reference.
    gate = numerical_gate(actual, expected)
    if actual.dtype == torch.bfloat16:
        rms = gate.get('reference_rms')
        if rms is not None:
            error = (actual.double()-expected).abs()
            gate.update(elementwise_tolerance_factor=.02, normalized_rmse_limit=.005,
                        passed=bool((error <= .02*(rms+expected.abs())+1e-6).all()) and
                               gate['normalized_rmse'] <= .005)
    gate['reference'] = REFERENCE
    return gate
