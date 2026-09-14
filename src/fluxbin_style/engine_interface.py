"""Project-owned inference interface reserved for future vLLM integration.

This module deliberately has no vLLM imports, registration or version coupling.
A future serving adapter owns weight loading/sharding, streams, workspace and
scheduler policy; the backend owns only layout conversion and the Linear op.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol
import torch
from .deployment import KERNELS, FORMAT, convert_artifact, m1_out, workspace_shape


@dataclass(frozen=True)
class LinearContract:
    format: str = FORMAT
    supported_m: tuple[int, ...] = (1,)
    input_dtypes: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
    accumulation_dtype: torch.dtype = torch.float32
    scale_dtype: torch.dtype = torch.float32
    group_size: int = 128
    sparse_columns: int = 8
    minimum_compute_capability: tuple[int, int] = (8, 0)
    fused_qkv: bool = False
    tensor_parallel: bool = False
    graph_capture_after_warmup: bool = True  # Intended contract; GPU validation pending.


class LinearBackend(Protocol):
    contract: LinearContract

    def convert(self, artifact: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    def workspace_shape(self, out_features: int, in_features: int) -> tuple[int, int]: ...
    def apply_out(self, x: torch.Tensor, layout: Mapping[str, torch.Tensor],
                  out: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor: ...


class M1Backend:
    contract = LinearContract()

    def __init__(self, groups_per_split=8, *, kernel="v1"):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.kernel=kernel
        if not isinstance(groups_per_split,int) or not 1<=groups_per_split<=1024:
            raise ValueError('groups_per_split must be 1..1024')
        self.groups_per_split=groups_per_split

    def convert(self, artifact):
        return convert_artifact(artifact)

    def workspace_shape(self, out_features, in_features):
        return workspace_shape(out_features,in_features,self.groups_per_split)

    def apply_out(self,x,layout,out,workspace):
        return m1_out(x,layout,out,workspace,groups_per_split=self.groups_per_split,kernel=self.kernel)
