# Marlin-Style FluxBin current handoff

## Current objective

Build and validate an independent two-base rank-one binary weight algorithm,
retaining a pure two-base arm and adding a sparse `s=8` residual-refinement arm.
Only after algorithm-quality acceptance should the project implement a
Marlin-style packed CUDA backend and measure real deployment speed.

## Frozen algorithm contract

- Exactly two binary bases in `{-1,+1}`.
- Group size 128 along the Linear input dimension.
- Each base has independent row and column rank-one scales.
- Exact joint search over four two-base sign patterns.
- FP32 optimization and saved algorithm scales.
- Retain the complete pure two-base arm as a separately reportable output.
- Hybrid-s8 selects 8 columns in every 128-column group by the pure arm's
  residual column SSE over output rows, using stable lower-index tie breaking.
- Store selected group-local indices in ascending order.
- Fit a second independent two-base rank-one decomposition only to the selected
  residual columns; non-selected columns must remain bit-exactly unchanged from
  the pure arm's reconstruction.
- No Shared-C, Hessian error propagation, calibration data, distillation, CUDA
  kernel, or serving integration.
- FluxBin is an algorithm-form source only. Its accuracy tables, calibration
  choices, kernel claims, and internally inconsistent statements are not
  project evidence or acceptance targets.

## Evidence gates

1. Local/static and synthetic algebra checks.
2. One real Qwen3-32B Linear pure two-base reconstruction on Isambard GH200.
3. One real Qwen3-32B Linear hybrid-s8 reconstruction on Isambard GH200.
4. Full-model reconstruction emitting pure two-base and hybrid-s8 together,
   manually authorized after gate 3 review.
5. Dense fake-quantized WikiText-2 PPL, manually authorized after gate 4.
6. Packed backend correctness and performance in separate later phases.

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

## Accepted pure two-base result

- Two-base reference implementation: revision
  `e94c751971e05c5dd1aa72723d8b59b8a59c2168`, pushed to `origin/main`.
- First real-Linear Slurm job `6249838` completed `0:0` on 2026-09-02 from that
  clean revision after all six synthetic tests passed.
- Greedy SSE: `2734.2239076773903`; optimized pure two-base SSE:
  `2325.262335431492` (14.9571354% reduction versus greedy).
- Optimized relative Frobenius error: `0.3381926`; cosine similarity:
  `0.9410769`; MSE: `5.54386e-5`.
- The solver reached the 50-iteration cap and was near a plateau, but did not
  meet its formal convergence threshold.
- Result SHA-256:
  `a8030dfcc7dc23a82825dc8d5b142211e63f2cd135e0c001e692dc0d52792ab8`.
- Payload SHA-256:
  `54d83c27214ba91a9b1ce3dbb60e234db19e741b803bb4c4414bec967732f37c`.
- Source-manifest SHA-256:
  `9f4d4678fd7b0f238c6cb3972a0549673807acacb683806494d3cf26b849cb6d`.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-v1/`.
- Logs:
  `logs/qwen3-32b-single-linear/fluxbin2-q32-linear-6249838.{out,err}` in the
  Isambard checkout.

## Prepared hybrid-s8 gate

- Parent: the hash-pinned accepted payload from job `6249838`; the global arm
  is loaded and independently reconstructed, not refitted.
- Target: the same Qwen3-32B `model.layers.0.self_attn.o_proj.weight` tensor.
- Payload retains global signs/scales and adds group-local `int16` indices plus
  refinement signs/scales.
- Required checks: accepted-parent hashes and inventory, target hash, finite
  metrics, strict SSE improvement versus pure two-base, unique/sorted indices,
  exact zero reconstruction delta outside selected columns, and payload
  round-trip hashes.
- The Slurm job runs all synthetic tests before loading the real tensor.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-s8-v1/`.

## Current status

- Local static checks for hybrid-s8: Python compilation, JSON validation,
  Slurm shell syntax, and `git diff --check` passed. The Mac has no project
  PyTorch environment, so tensor tests remain an Isambard preflight gate.
- Hybrid-s8 real-Linear Slurm job `6250990` was submitted on 2026-09-02 from
  clean revision `33d559091350d14d0510102cda6e0b4151874dba`; last bounded
  observation was `PENDING` with no failure reason immediately after submission.
- Hybrid-s8 logs:
  `logs/qwen3-32b-single-linear-s8/fluxbin2-s8-q32-6250990.{out,err}` in the
  Isambard checkout.
- Full-model reconstruction: not launched.
- PPL: not launched.
- Backend: not implemented.
