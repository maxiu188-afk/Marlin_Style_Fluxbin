# Qwen3-8B algorithm-quality and deployment plan

2026-09-14 execution order: start with M=1 on Qwen3-8B, then a single complete
transformer block, then full-model expansion. M>1 and 32B stress cells are later
extensions, not part of the first trial. Reserve an engine-neutral interface for
future vLLM; do not integrate it now. Current work is offline preparation only.
First new server startup MUST record the environment before and after setup;
after successful CUDA smoke, prepare an immutable reusable image and validate it
on a recreated Pod. See `M1_ACCELERATION_PREPARATION.md`.

Status (2026-09-13, latest user direction): accuracy experiments pause after the
accepted fixed step400 test PPL, **13.169788495**, improved from undistilled
conditioned hybrid 14.951611048. The persistent step400 artifact is the frozen
starting point for acceleration research. The user explicitly prioritizes
performance and authorizes research correctness/benchmark work despite the
remaining 35.4228% gap over BF16. This supersedes the earlier sequencing rule
that required passing the 5% gate before any performance engineering.
Original quality thresholds/results remain reported; no claim of product
quality acceptance or measured acceleration is made. See
`QWEN3_8B_DISTILLED_TEST_RESULTS.md` and `ACCELERATION_HANDOFF.md`.

RunPod image, storage, Template, cache, and restart conventions are defined in
[`infra/runpod/README.md`](../infra/runpod/README.md). The reusable environment is
part of Phase A0. The user now prefers an existing RunPod PyTorch/CUDA template:
inspect the provided environment and add only missing dependencies. Record its
base image and actual versions; a custom image is optional later, not an A0
prerequisite. Freeze the accepted environment before matched quality runs.

## Objective

Move the active full-model target from Qwen3-32B to Qwen3-8B and answer two
questions, with quality improvements now paused by user direction:

1. Can the calibrated two-base representation produce a complete, reproducible
   Qwen3-8B quantized checkpoint with acceptable model quality?
2. Using the fixed accepted step400 artifact, can a packed implementation provide useful
   operator, block, full-model, and serving acceleration on real NVIDIA GPUs?

Qwen3-32B is no longer an active full-model target. Its accepted quality results
remain historical evidence, and its exact Linear shapes remain useful as
operator-only stress cases. No new 32B reconstruction, PPL, or serving run is
part of this plan.

## Frozen scope

- Primary model: `Qwen/Qwen3-8B`.
- Model revision: reuse the previously accepted pinned revision
  `b968826d9c46dd6066d109eabc6255188de91218` after a preflight verifies that the
  snapshot, tokenizer, model structure, and expected tensor inventory match.
- Expected transformer scope: 36 layers, seven target Linears per layer, 252
  target weights, and 6,945,767,424 included weight elements. Embeddings,
  normalization parameters, biases, and output head remain outside the
  quantization contract unless a later written revision says otherwise.
- Algorithm arms: BF16 reference, pure two-base Hessian-OBQ, and hybrid-s8
  Hessian-OBQ.
- Preserve the accepted algorithm choices while changing only the model:
  two binary bases, group size 128, exact four-pattern assignment, independent
  row/column scales, 50-step ALS limit, C4 256x2048 calibration with seed
  `20260902`, scalar-normalized `H = 2 X^T X`, and 1% mean-diagonal damping.
- Primary workload: autoregressive decode. Decode batch/token-row sizes
  `M = 1, 2, 4, 8` are reported separately. Prefill is outside the initial
  claim and must not be inferred from decode results.
- No phase automatically launches its successor.

Changing the model is the experimental variable. Algorithm changes such as
additional bases, Shared-C, distillation, a different residual budget, or a new
calibration protocol require a separately versioned arm rather than silently
changing this baseline.

## Phase A: algorithm and quantized-weight production

### A0. Port and preflight

Before any real-model execution:

- select an existing RunPod PyTorch/CUDA template, inspect its environment,
  add only necessary dependencies, verify persistent storage, and record the
  base image/digest and actual environment versions; custom image publication
  is optional after the required stack is established;
- remove Qwen3-32B-only assumptions from the new versioned configs and runners
  without changing historical 32B configs or results;
- verify the pinned model revision, 36-layer / 252-Linear inventory, tensor
  names, shapes, dtypes, tokenizer, and calibration compatibility;
- materialize a new hash-pinned calibration artifact or prove exact reuse of an
  existing artifact under the same token protocol;
- run the existing synthetic Hessian, OBQ, packing, materialization, and resume
  tests locally;
- record source, config, model, tokenizer, and calibration hashes.

Acceptance: all static and synthetic checks pass, the real-model inventory is
exact, and no 32B path or artifact can be overwritten by the 8B run.

### A1. Representative real-Linears

Run the pure and hybrid arms on a small shape set containing:

- the accepted `o_proj [4096,4096]` reference shape;
- at least one attention projection shape;
- the largest Qwen3-8B MLP expansion and contraction shapes found by A0.

For every cell, require finite metrics, deterministic tensor inventories,
valid packed fields, exact zero outside hybrid-selected columns, and strict
hybrid reconstruction improvement over its matched pure parent. Record wall
time and peak memory as planning data, not deployment acceleration.

Acceptance: all representative shapes pass the algorithm contract. A failure
returns to algorithm diagnosis; it does not authorize a full-model run.

### A2. Complete Qwen3-8B quantization

Produce separately resumable pure and hybrid artifacts for all 36 layers and
252 target Linears. Each branch must propagate its own quantized hidden states
and Hessians exactly as in the accepted 32B protocol.

Required outputs:

- one immutable metadata/payload pair per layer and arm;
- complete tensor inventories, shapes, dtypes, internal hashes, and source
  manifests;
- aggregate and per-module weight SSE, relative Frobenius error, propagated
  output-SSE diagnostics, elapsed time, and peak memory;
- an explicit logical-bpw estimate separated from physical experiment storage;
- a portable manifest sufficient to reconstruct or transfer the accepted
  quantized checkpoint.

Acceptance: 36 contiguous layers and all 252 Linears are present for each arm;
all metrics are finite; all hashes and inventories pass; the hybrid arm meets
the declared per-tensor comparison contract. Scheduler completion alone is not
acceptance.

### A3. Matched model-quality evaluation

Evaluate BF16, pure, and hybrid in the same run on the frozen WikiText-2
protocol: 146 non-overlapping 2048-token blocks and 298,862 scored transitions.
Each quantized arm must validate every payload and internal hash before dense
materialization.

Primary quality gate:

- the candidate for deployment must have finite PPL on exactly the same scored
  transitions as BF16;
- relative PPL gap versus matched BF16 must be at most 5%;
- the historical quality gate remains a model-quality criterion;
- the latest user authorization permits research acceleration work on the fixed
  step400 artifact while further quality experiments are paused, even though
  the recorded gap remains above 10%. This does not certify deployment quality.

The 5% and 10% boundaries are project acceptance choices, not claims from the
paper. Reconstruction improvements cannot substitute for the PPL gate.

Optional downstream task-accuracy evaluation may be added later under its own
frozen datasets and metrics. It is not automatically launched by the PPL run.

## Phase B: real deployment and acceleration

Phase B uses the exact step400 artifact with accepted integrity/reconstruction
and test PPL provenance. Its manifest is pinned on the persistent volume; the
latest user direction permits performance research without further quality tuning.

### B0. Runtime contract and packed layout

- define a versioned deployment layout separately from the algorithm-artifact
  format;
- implement a CPU/PyTorch reference decoder for the deployment layout;
- convert the accepted algorithm payload without refitting or changing its
  quantized values;
- hash the source artifact and converted payload so conversion is auditable;
- define accumulation precision, scale application, padding, supported shapes,
  and fallback behavior.

Acceptance: bit fields decode exactly and reference outputs agree with dense
materialization under predeclared tolerances. This phase makes no speed claim.

### B1. CUDA kernel correctness

Implement and compile architecture-specific paths rather than assuming one
binary is representative everywhere:

- SM90 development and tuning on H20;
- formal Hopper validation on H200;
- formal SM80 validation on A100 80GB.

Correctness covers every Qwen3-8B Linear shape, `M = 1, 2, 4, 8`, edge/padding
cases, deterministic repeated execution, finite outputs, and the exact accepted
payload. Unsupported cells must take an explicit correct fallback.

No timing result is reportable until the same cell passes correctness.

### B2. Operator performance

Measure same-run latency for every real Qwen3-8B Linear shape and each decode
`M`. Report warmup, repetitions, timing mechanism, clocks/power policy when
available, input reuse, CUDA Graph versus eager mode, and stability statistics.

Required baselines:

- same-run dense BF16/FP16 execution;
- the project's unfused or reference packed path;
- one mature low-bit backend when an apples-to-apples compatible route exists.

Also measure the retained Qwen3-32B shape suite as synthetic operator-only
stress cases. These rows test shape scaling; they do not constitute a 32B
full-model or serving claim.

### B3. Block integration

Integrate the accepted kernel into representative attention and MLP blocks.
Measure the complete block, including activation preparation, quantization,
packing, scale application, kernel launches, reductions, and fallback work.

Acceptance: block outputs pass the declared numerical gate, and timing is
reported independently from isolated Linear timing. Operator speedups must not
be multiplied to estimate a block result.

### B4. Full-model and serving evaluation

Load the complete accepted Qwen3-8B packed checkpoint through the selected
runtime and evaluate real autoregressive generation.

Report at minimum:

- cold startup and checkpoint-load time;
- peak device memory and packed weight bytes;
- time to first token separately from decode latency;
- inter-token latency and tokens/s for `M = 1, 2, 4, 8`;
- eager and CUDA-Graph results separately when both are supported;
- identical prompts, generation settings, output/token checks, and same-run
  BF16/FP16 baselines;
- kernel coverage, fallback counts, and time outside quantized Linears.

Formal final tables use H200 and A100 80GB. H20 is the iterative SM90
development platform and may also be reported, but it does not replace the
formal H200 result. Hardware, software, clocks, framework version, and source
revision must be recorded for each platform.

Acceptance requires both numerical/token correctness and stable repeated
timing. A fast run that changes tokens or log probabilities beyond the frozen
gate is rejected. Full-model and serving speedups must be measured directly;
they cannot be inferred from operator results.

## Decision points and stopping rules

1. Stop before full quantization if representative real-Linears violate the
   algorithm contract.
2. Preserve the failed model-quality decision; under the latest user direction,
   research correctness/benchmark work may proceed on the fixed step400 artifact.
   Product deployment acceptance remains separate.
3. Stop performance promotion for any shape or batch cell that fails numerical
   correctness or timing stability.
4. If 8B full-model acceleration is weak but 32B stress shapes are strong,
   report a shape-dependent operator crossover. Do not restart a 32B full-model
   program automatically.
5. Consider a 14B-class model only after the 8B shape/performance curve shows a
   concrete size limitation and a separate plan pins the model and protocol.

## Evidence hierarchy

Keep the following claims independent:

1. algorithm reconstruction correctness;
2. complete quantized-weight artifact;
3. dense fake-quant model quality;
4. deployment-layout conversion correctness;
5. CUDA operator correctness and performance;
6. block correctness and performance;
7. full-model latency, throughput, and memory;
8. serving behavior.

Passing an earlier level authorizes work on the next level but does not count as
evidence for it.
