# Marlin-Style FluxBin current handoff

## Current objective

Build and validate an independent two-base rank-one binary weight algorithm.
Only after algorithm-quality acceptance should the project implement a
Marlin-style packed CUDA backend and measure real deployment speed.

## Frozen phase-1 contract

- Exactly two binary bases in `{-1,+1}`.
- Group size 128 along the Linear input dimension.
- Each base has independent row and column rank-one scales.
- Exact joint search over four two-base sign patterns.
- FP32 optimization and saved algorithm scales.
- No Shared-C, salient refinement, compensation matrix, Hessian error
  propagation, distillation, CUDA kernel, or serving integration.
- FluxBin is an algorithm-form source only. Its accuracy tables, calibration
  choices, kernel claims, and internally inconsistent statements are not
  project evidence or acceptance targets.

## Evidence gates

1. Local/static and synthetic algebra checks.
2. One real Qwen3-32B Linear reconstruction on Isambard GH200.
3. Full-model reconstruction, manually authorized after gate 2 review.
4. Dense fake-quantized WikiText-2 PPL, manually authorized after gate 3.
5. Packed backend correctness and performance in separate later phases.

No runner automatically launches its successor.

## First real-Linear gate

- Model: `Qwen/Qwen3-32B`.
- Snapshot: `9216db5781bf21249d130ec9da846c4624c16137`.
- Tensor: `model.layers.0.self_attn.o_proj.weight`.
- Shape: `[5120,8192]`.
- Tensor SHA-256: `af8343c70597ac417bf2daa550282fdc9639a23b3496506df99d9f168fc6adec`.
- Comparison: greedy two-base row-scale initialization versus optimized
  independent rank-one row/column factors with joint sign reassignment.
- Required execution gates: pinned tensor/runtime, finite metrics, monotonic
  scale and assignment stages, payload round-trip hashes, and no automatic
  follow-up.

The retained 32B snapshot remains under the historical QBB-New artifact root
and must not be modified or duplicated:

```text
${PROJECTDIR}/${USER}/qbb-new/huggingface/hub/
  models--Qwen--Qwen3-32B/snapshots/
  9216db5781bf21249d130ec9da846c4624c16137/
```

## Environment boundary

- macOS/Apple Silicon: source review, static compilation, and small CPU tests
  when a local project environment exists.
- Isambard GH200: real weights, reconstruction, PPL, and future performance.
- Submit durable work through Slurm and perform one bounded startup check.
- Scheduler `COMPLETED` is not acceptance; review structured JSON, provenance,
  hashes, payload inventory, finite metrics, and the gate contract.

## Current status

- Two-base reference implementation: prepared locally, validation pending.
- Synthetic tests: not yet executed because the Mac has no project PyTorch
  environment; the Slurm gate runs them before reading the real tensor.
- First real-Linear Slurm job: not yet submitted.
- Full-model reconstruction: not launched.
- PPL: not launched.
- Compensation branch: not implemented.
- Backend: not implemented.
