"""Layer-sequential calibration helpers for Qwen3 quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fluxbin_style.hessian_obq import InputHessianAccumulator


class CaptureComplete(Exception):
    """Control-flow signal used to stop after the first decoder layer input."""


def hidden_states(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, "hidden_states", None)
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError("decoder layer returned unsupported output")


@dataclass(frozen=True)
class FirstLayerCapture:
    inputs: torch.Tensor
    forward_kwargs: dict[str, Any]


@torch.no_grad()
def capture_first_layer_inputs(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    device: torch.device,
) -> FirstLayerCapture:
    """Capture fixed-length decoder inputs without executing the layer stack."""

    if tokens.ndim != 2 or tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
        raise ValueError("tokens must have shape [samples, sequence_length]")
    layers = model.model.layers
    samples, sequence_length = tokens.shape
    dtype = next(model.parameters()).dtype
    inputs = torch.empty(
        (samples, sequence_length, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    forward_kwargs: dict[str, Any] = {}
    captured = 0

    class Catcher(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, value: torch.Tensor, **kwargs: Any):
            nonlocal captured
            if captured >= samples:
                raise RuntimeError("captured more calibration sequences than expected")
            inputs[captured].copy_(value[0])
            if not forward_kwargs:
                forward_kwargs.update(kwargs)
            captured += 1
            raise CaptureComplete()

    original = layers[0]
    previous_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    layers[0] = Catcher(original)
    try:
        for sample_index in range(samples):
            try:
                model(
                    input_ids=tokens[sample_index : sample_index + 1].to(device),
                    use_cache=False,
                )
            except CaptureComplete:
                pass
    finally:
        layers[0] = original
        model.config.use_cache = previous_cache
    if captured != samples:
        raise RuntimeError("failed to capture every calibration sequence")
    return FirstLayerCapture(inputs=inputs, forward_kwargs=forward_kwargs)


def qwen3_hessian_sources(layer: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """Return one representative for each exact-input Linear group."""

    sources = {
        "qkv": layer.self_attn.q_proj,
        "o": layer.self_attn.o_proj,
        "gate_up": layer.mlp.gate_proj,
        "down": layer.mlp.down_proj,
    }
    if any(not isinstance(module, torch.nn.Linear) for module in sources.values()):
        raise TypeError("Qwen3 Hessian source is not Linear")
    return sources


@dataclass(frozen=True)
class LayerHessianCapture:
    hessians: dict[str, torch.Tensor]
    activation_rows: dict[str, int]


@dataclass(frozen=True)
class LinearHessianCapture:
    hessian: torch.Tensor
    activation_rows: int


@torch.no_grad()
def capture_linear_hessian(
    layer: torch.nn.Module,
    source: torch.nn.Linear,
    inputs: torch.Tensor,
    forward_kwargs: dict[str, Any],
) -> LinearHessianCapture:
    """Capture one exact-input Hessian after earlier modules were replaced.

    Re-running the layer between true-sequential groups is deliberate: the
    source input then reflects all already-quantized modules in that layer.
    """

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [samples, sequence_length, hidden]")
    if not isinstance(source, torch.nn.Linear):
        raise TypeError("source must be Linear")
    accumulator = InputHessianAccumulator(source.in_features, device=inputs.device)

    def collect(
        _module: torch.nn.Module,
        module_inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        accumulator.add(module_inputs[0])

    handle = source.register_forward_hook(collect)
    try:
        for sample_index in range(inputs.shape[0]):
            hidden_states(layer(inputs[sample_index : sample_index + 1], **forward_kwargs))
    finally:
        handle.remove()
    expected_rows = inputs.shape[0] * inputs.shape[1]
    if accumulator.sample_count != expected_rows:
        raise RuntimeError("activation-row count drifted")
    return LinearHessianCapture(
        hessian=accumulator.value(),
        activation_rows=accumulator.sample_count,
    )


@torch.no_grad()
def capture_layer_hessians(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    forward_kwargs: dict[str, Any],
) -> LayerHessianCapture:
    """Capture qkv, o, gate/up, and down Hessians in one layer pass."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [samples, sequence_length, hidden]")
    sources = qwen3_hessian_sources(layer)
    accumulators = {
        name: InputHessianAccumulator(module.in_features, device=inputs.device)
        for name, module in sources.items()
    }
    handles = []
    for name, module in sources.items():
        def collect(
            _module: torch.nn.Module,
            module_inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            *,
            name: str = name,
        ) -> None:
            accumulators[name].add(module_inputs[0])

        handles.append(module.register_forward_hook(collect))
    try:
        for sample_index in range(inputs.shape[0]):
            hidden_states(layer(inputs[sample_index : sample_index + 1], **forward_kwargs))
    finally:
        for handle in handles:
            handle.remove()
    expected_rows = inputs.shape[0] * inputs.shape[1]
    for name, accumulator in accumulators.items():
        if accumulator.sample_count != expected_rows:
            raise RuntimeError(f"activation-row count drifted for {name}")
    return LayerHessianCapture(
        hessians={name: accumulator.value() for name, accumulator in accumulators.items()},
        activation_rows={name: accumulator.sample_count for name, accumulator in accumulators.items()},
    )


@torch.no_grad()
def propagate_layer_inputs(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    forward_kwargs: dict[str, Any],
) -> None:
    """Run one fully quantized decoder layer into a preallocated buffer."""

    if inputs.shape != outputs.shape or inputs.device != outputs.device:
        raise ValueError("input and output buffers must match")
    for sample_index in range(inputs.shape[0]):
        value = hidden_states(
            layer(inputs[sample_index : sample_index + 1], **forward_kwargs)
        )
        outputs[sample_index].copy_(value[0])
