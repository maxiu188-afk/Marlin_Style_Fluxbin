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

## Submitted pure two-base 50-versus-200 convergence check

- The accepted job `6249838` exhausted all 50 iterations with stop reason
  `max_iters`; its final iteration still reported relative improvement
  `2.3623791e-6`, above the solver threshold `1e-6`.
- The diagnostic keeps the same target tensor/hash, seed, initializer,
  algorithm, grouping, FP32 scale math, convergence patience, tolerance, and
  assignment chunking. Only `max_iters` changes from 50 to 200.
- The accepted baseline JSON is bound by SHA-256
  `a8030dfcc7dc23a82825dc8d5b142211e63f2cd135e0c001e692dc0d52792ab8`.
- “50 steps was too small” means the 200-step final SSE is below the 50-step
  final SSE by more than the frozen `1e-6` relative comparison tolerance. A
  negative or very small effect remains a valid diagnostic result.
- Job `6261256` was submitted on 2026-09-02 from clean revision
  `3e641e5be1213fcb5fd8daea122d93b2617c9b39`. The server preflight passed all
  15 unit tests, Python compilation, Slurm syntax, and the baseline-result hash
  and iteration-history checks. The bounded startup snapshot observed
  `PENDING (Priority)` with 1 GH200 GPU, 8 CPUs, and a 15-minute limit.
- It uses the isolated server checkout
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/repo-single-linear-i200` so the
  already queued PPL job keeps its original source checkout unchanged.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-max-iters-200-v1/`.
- Logs:
  `logs/qwen3-32b-single-linear-i200/fluxbin2-q32-i200-6261256.{out,err}` in
  the isolated checkout.
- Acceptance and the answer to the 50-step question remain pending structured
  result review; the scheduler state alone is not evidence.
- No full-model, PPL, or backend follow-up is launched automatically.

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

## Accepted hybrid-s8 single-Linear result

- Job `6250990` completed the application path and all 9 tests; SlurmDBD did
  not expose its scheduler accounting record, so scheduler `State/ExitCode`
  remains independently unavailable.
- Pure global SSE: `2325.262335431492`; hybrid-s8 SSE:
  `2142.084815168282`, a `7.8777142%` reduction.
- Relative Frobenius error: `0.3381925903` -> `0.3245984495`.
- Selected columns contained `8.3860997%` of global residual energy and the
  refinement removed `93.9377595%` of their SSE.
- Result SHA-256:
  `c4eb7effe1ad89a05689f8abbc713d65392c78fea68a699d20d72bdb0de31390`.
- Payload SHA-256:
  `bc1e69b788d4cdccaf8c5f186fe32d87182ab4d8c7a7859b763bba2b6bf59fee`.

## Full-model reconstruction contract

- Scope: all 448 transformer-block Linears across 64 Qwen3-32B layers,
  totalling 31,205,621,760 weights. Embeddings, output head and non-matrix
  tensors are excluded by a fail-closed inventory.
- Each Linear computes both the pure global two-base arm and hybrid-s8 in one
  pass. The hybrid reuses that exact global payload and stores only refinement
  signs/scales/indices in addition.
- One metadata JSON and one payload per Linear provide bounded restart. Resume
  requires matching target, config, implementation and payload hashes.
- Both global and refinement signs are losslessly stored in
  `fluxbin-two-base-interleaved-2bit-v1`. This is an algorithm artifact format,
  not a CUDA backend format or performance claim.
- Expected packed payload is about 12,222,480,384 bytes (11.38 GiB), excluding
  small safetensors/JSON overhead. Aggregate algorithm cost is about 2.5078 bpw
  for pure global and 3.1334 bpw for hybrid-s8 with FP32 scales.
- Required gates include 448/448 inventory, exact single-Linear reproduction,
  strict hybrid SSE improvement for every tensor, zero delta outside selected
  columns, packed-sign round trips, finite/monotonic solver records, complete
  896-file artifact inventory, and aggregate metrics.
- PPL and backend jobs are not launched automatically.

## Current status

- Local static checks for hybrid-s8: Python compilation, JSON validation,
  Slurm shell syntax, and `git diff --check` passed. The Mac has no project
  PyTorch environment, so tensor tests remain an Isambard preflight gate.
- Hybrid-s8 real-Linear job `6250990`: application-level accepted as documented
  above.
- Hybrid-s8 logs:
  `logs/qwen3-32b-single-linear-s8/fluxbin2-s8-q32-6250990.{out,err}` in the
  Isambard checkout.
- Full-model reconstruction job `6253233` was submitted on 2026-09-02 from
  clean revision `5234079df1807cab8b9c8633070c9fe78a17c963`; the bounded
  startup check observed `RUNNING` on `nid010224` after config, GH200 runtime,
  and CUDA packed-sign round-trip preflight passed. It was entering the 14-test
  stage at the last observation.
- Its time limit was reduced from 16 hours to 12 hours after submission. The
  closest QBB-New 448-Linear Qwen3-32B reconstruction took
  `30897.34031198197` seconds (8h34m57s), so 12 hours keeps about 40% margin
  while avoiding an unnecessarily long resource request. The checked end time
  is `2026-09-03T01:46:28+00:00`.
- Full-model result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-full-two-base-rank1-s8-v1/`.
- Full-model resumable artifacts:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/artifacts/qwen3-32b-full-two-base-rank1-s8-v1/decompositions/`.
- Full-model logs:
  `logs/qwen3-32b-full-two-base-rank1-s8-v1/fluxbin-full-q32-s8-6253233.{out,err}`.
- Full-model application and independent artifact audit accepted: 448/448
  tensors strictly improved, 896/896 files and all packed decoded-sign hashes
  passed. Aggregate pure/hybrid SSE is
  `1907722.585722923/1762887.8386713415` (`7.5920235%` lower); relative
  Frobenius error is `0.3406372032/0.3274513676`.
- Full-model result SHA-256:
  `bf412e333e7ffe761c0627fecc6b2159ae19faa60f3dc48a125713fcd802084c`.
- Full-model source-manifest SHA-256:
  `6c418d7453565a784ecf3f2d83e9dd4a5cbe2abbb546e6de539d373538cac06a`.
- PPL gate is prepared for matched BF16, pure global two-base, and hybrid-s8.
  It reuses the accepted QBB-New protocol of 146 non-overlapping 2048-token
  blocks and 298,862 scored transitions, and binds BF16 reference PPL
  `7.6108390395557874` from job `6154681`.
- PPL materialization is dense BF16 fake quantization from the packed algorithm
  payload; it is not packed backend or speed evidence.
- PPL job `6259037` was submitted on 2026-09-02 from clean revision
  `59fe6e7080eee720321c653fdc6102f4c6a5d5e1`. Server preflight passed all
  15 unit tests, Python compilation, Slurm shell syntax, and the previously
  recorded real-protocol validation. The one bounded startup snapshot observed
  `PENDING (Priority)` with 1 GH200 GPU, 16 CPUs, and a 1-hour time limit.
- PPL result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-two-base-rank1-s8-ppl-v1/`.
- PPL logs:
  `logs/qwen3-32b-two-base-rank1-s8-ppl/fluxbin-ppl-q32-s8-6259037.{out,err}`.
- PPL acceptance remains pending; scheduler state alone is not evidence that
  the three PPL arms completed or passed their provenance and metric gates.
- Backend: not implemented.
