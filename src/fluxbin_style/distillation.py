"""Scale-only hybrid distillation; QBB-New loss semantics, packed fixed buffers."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .evaluation import materialize_hybrid_s8_weight

SCALES = ('global_row_scales', 'global_column_scales',
          'refinement_row_scales', 'refinement_column_scales')
FIXED = ('global_sign_codes', 'refinement_indices', 'refinement_sign_codes')


class HybridScaleLinear(nn.Module):
    """Recompute dense weights during backward; retain packed signs and FP32 scales."""
    def __init__(self, payload, *, group_size=128, columns_per_group=8, bias=None):
        super().__init__()
        if set(payload) != set(SCALES + FIXED):
            raise ValueError('hybrid payload inventory drifted')
        self.group_size, self.columns_per_group = group_size, columns_per_group
        for name in FIXED:
            self.register_buffer(name, payload[name].detach().clone())
        for name in SCALES:
            self.register_parameter(name, nn.Parameter(payload[name].detach().float().clone()))
        self.register_buffer('bias', None if bias is None else bias.detach().clone())
        self.out_features = payload['global_sign_codes'].shape[0]
        self.in_features = payload['global_sign_codes'].shape[1] * 4
        # The accepted decoder performs shape/range/finite validation.
        with torch.no_grad():
            self.reconstruct_weight(torch.bfloat16)

    def reconstruct_weight(self, dtype):
        return materialize_hybrid_s8_weight(
            **{name: getattr(self, name) for name in SCALES + FIXED},
            group_size=self.group_size, columns_per_group=self.columns_per_group,
            device=self.global_row_scales.device, output_dtype=dtype)

    def forward(self, inputs):
        def linear(x):
            return F.linear(x, self.reconstruct_weight(x.dtype),
                            None if self.bias is None else self.bias.to(x.dtype))
        if torch.is_grad_enabled():
            return checkpoint(linear, inputs, use_reentrant=False)
        return linear(inputs)

    def export_payload(self):
        return {name: getattr(self, name).detach().cpu().contiguous().clone()
                for name in SCALES + FIXED}


def distillation_loss(logits, input_ids, teacher_features, student_features,
                      normalizers, *, expected_layers=36):
    """Hard next-token CE plus mean block-output MSE; fixed initial normalizers.

    Inputs are equal-length unpadded sequences, as in QBB-New sample200.
    Teacher features are explicitly detached, even if the caller forgot no_grad.
    """
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or input_ids.shape[1] < 2:
        raise ValueError('expected unpadded [batch, sequence>=2] inputs and logits')
    if len(teacher_features) != expected_layers or len(student_features) != expected_layers or expected_layers < 1:
        raise ValueError('feature layer coverage drifted')
    if set(normalizers) != {'ce', 'feature'} or not all(
        math.isfinite(float(v)) and float(v) > 0 for v in normalizers.values()
    ):
        raise ValueError('fixed initial normalizers must be finite and positive')
    terms = []
    for teacher, student in zip(teacher_features, student_features, strict=True):
        if teacher.shape != student.shape or student.ndim != 3 or student.shape[:2] != input_ids.shape:
            raise ValueError('feature shape drifted')
        terms.append(F.mse_loss(student.float(), teacher.detach().float()))
    feature = torch.stack(terms).mean()
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         input_ids[:, 1:].reshape(-1))
    total = ce / float(normalizers['ce']) + feature / float(normalizers['feature'])
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite distillation loss')
    return {'total': total, 'ce': ce, 'feature': feature}


def forward_losses(teacher, student, input_ids, normalizers, *, expected_layers=36):
    """Capture post-block residual-stream outputs; remove hooks even on failure."""
    teacher_features, student_features, handles = [], [], []
    def capture(target):
        def hook(_module, _inputs, output):
            target.append(output[0] if isinstance(output, tuple) else output)
        return hook
    if len(teacher.model.layers) != expected_layers or len(student.model.layers) != expected_layers:
        raise ValueError('teacher/student architecture drifted')
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError('teacher must be frozen and eval')
    try:
        for model, features in ((teacher, teacher_features), (student, student_features)):
            for block in model.model.layers:
                handles.append(block.register_forward_hook(capture(features)))
        with torch.no_grad():
            teacher(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        output = student(input_ids=input_ids, use_cache=False, logits_to_keep=0)
        return distillation_loss(output.logits, input_ids, teacher_features, student_features,
                                 normalizers, expected_layers=expected_layers)
    finally:
        for handle in handles:
            handle.remove()
