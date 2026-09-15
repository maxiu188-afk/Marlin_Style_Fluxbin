# Marlin-Style FluxBin current handoff

Latest GPU result (2026-09-15, A100 SXM4 80GB, prepared v2.1): full-model
32-step sequence Graph achieved **1.381x / 1.383x original BF16 speed**, with
both prompts stable (all Graph timing spans <0.12%). Eager timing was unstable
at the 5% threshold, so aggregate status is `completed_unstable`. All 113 GPU
tests passed; checked/prepared wrappers and Graph outputs were exact in all six
arm/prompt cells. Dynamic/static attention and packed/decoded numerical differences
remain report-only. Evidence downloaded/hash-verified; GPU processes exited.
See [full-model results](QWEN3_8B_M1_LINEAR_RESULTS.md). Earlier pending statements below are historical.

Local preparation (2026-09-15): prepared full-model v2 adds bound packed Linear buffers, real-prefix static KV, interleaved 8-warmup/10-repeat timing and a complete 32-step CUDA Graph comparison. Original BF16 remains the primary baseline. CUDA validation and performance are pending; historical v1 results are unchanged. See [protocol and launch commands](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md).

Latest GPU result (2026-09-15, A100 80GB PCIe): 106 tests passed, 168/168
Linear numerical cells passed, 154/168 timing cells stable. Selected
`v5_p1024/gps1` achieved 2.1285x isolated Graph / 2.0219x isolated eager speed
(seven Linear time sums), but full-model speed was only 0.87696x / 0.86674x
original BF16. All full-model timings were stable; numerical differences remain
report-only. Block dense timing was unstable. Evidence downloaded/hash-verified;
no experiment processes remain. See [latest results](QWEN3_8B_M1_LINEAR_RESULTS.md).

Latest v4 GPU result (2026-09-15): A100 SXM4, 98 tests passed; candidate numerical
checks 84/84, stable timing cells 80/84. Selected v4/gps4 did not beat v3 or dense.
Full-model report-only completed in 119s with both prompts stable: 0.83063x /
0.83143x original BF16 speed, no acceleration. Unstable block timing was explicitly
allowed without changing numerical/route/provenance gates. Raw evidence is backed
up and hash-verified; no experiment processes remain. Persistent venv reuse took
36.9s (30.2s imports), versus local imports 2.8–3.1s; trials used local venv.
See [v4 results](QWEN3_8B_M1_LINEAR_RESULTS.md). Earlier local/pending statements are historical.

Local v4 implementation: column-scaled activations are prepared once per group,
then sign dot products are row-scaled, with eight sparse positions handled separately.
The new structural FP64 Linear reference is independent of legacy BF16 weights;
all three launches are timed. CUDA compilation/performance remain unvalidated.
See [v4 contract](M1_V2_LOCAL_OPTIMIZATION.md).

Latest direction (2026-09-15): user cancelled diagnostic preparation/execution.
Prioritize a candidate that factors column scales into activations and applies row
scales after sign dot products, with a separate sparse contribution. Static cost
estimates are hypotheses, not measured bottleneck attribution. Existing numerical
references/results remain intact; a new candidate needs an explicit reference.
See [design and evidence boundaries](M1_V2_LOCAL_OPTIMIZATION.md).

Earlier local preparation (retained, paused): `profile_m1_bottleneck.py` provides bounded
real-weight timing and profiler capture; `prepare_persistent_runtime.py` prepares
fingerprinted persistent Linux venv reuse. Neither has run on the GPU server yet.
Local suite: 93 tests, 90 passed and 3 CUDA skips. Commands and diagnostic limits:
[runbook](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md).

Latest GPU trial (2026-09-15): v3 compiled on A100 80GB PCIe; all 91 tests passed.
The fixed candidate batch passed 84/84 numerical cells (67/84 stable timings).
Selected v3/gps4 improved Linear timing over v1/v2 but remained slower than dense.
Full-model report-only run exited 0 in 312s: prompt 0 achieved 0.86887x original BF16
speed; prompt 1 has an unstable auxiliary decoded baseline, so overall status is
`completed_unstable`. No full-model speedup. Evidence downloaded/hash-verified;
no experiment processes remain. See [v3 results](QWEN3_8B_M1_LINEAR_RESULTS.md).
Earlier preparation and SXM4 statements below are historical.

Local kernel follow-up: explicit `v3` now implements three-stage cp.async staging,
register fragment double buffering and FP16/BF16 Tensor Core MMA for hybrid M=1.
It retains the v1 layout and a deterministic split reduction, with a direct-store
single-split path. This is implementation preparation only: no NVCC/GPU validation
or speed result yet. See [kernel design](M1_V2_LOCAL_OPTIMIZATION.md) and the new fixed Marlin candidate config.

Shutdown readiness (2026-09-14): live check found no experiment/tmux/GPU compute
processes. All 68 small-evidence files were archived, downloaded and hash-verified.
Keep network volume `34au39ljvf`; user may close compute. Actual shutdown has not
been confirmed and was not performed by the agent. See [shutdown record](SERVER_SHUTDOWN_READY.md).

Latest completed full-model trial (2026-09-14): report-only run exited 0 in 135s.
On A100 SXM4, packed decode is 0.9332x / 0.9562x the original BF16 speed on the two
fixed prompts (latency +7.16% / +4.58%); all decode timings and 252-Linear route
checks passed. Numerical differences were retained under the user-authorized
report-only policy. No full-model speedup was achieved. Details: [full-model results](QWEN3_8B_M1_LINEAR_RESULTS.md).

User follow-up: proceed with full-model timing despite finite numerical differences.
Runner now has explicit `--numerical-policy report-only`; errors remain recorded,
coverage/context/nonfinite guards remain active, and original BF16 is the primary
performance baseline. This does not relabel the failed strict trial as passed.

Full-model follow-up: the submitted task exited 1 at packed_step400 prompt 0,
repeat 0. Coverage (252 Linears) and KV length passed, but the same-weight decoded
BF16 correctness comparison failed: NRMSE 0.0115685 > 0.005, maximum logprob error
0.3394165 > 0.05, and 1/33 greedy predictions differed. No accepted full-model
speedup is available. This supersedes the earlier observed-running status.

Latest GPU evidence (2026-09-14, A100 SXM4): all 85 tests passed; bounded candidate
batch passed correctness in 84/84 cells, with 77/84 stable timings. The selected
v2/gps8 block passed correctness/stability but took 1213.783 us vs decoded BF16
1049.944 us (0.8650x speedup). User explicitly requested full-model measurement
regardless of block speed; full-model task `m1-sxm4-full` was submitted and observed
running, not accepted. See [SXM4 results](QWEN3_8B_M1_LINEAR_RESULTS.md).
Earlier unlaunched/preparation statements below are historical.

Offline preparation follow-up: candidate batch summarizer now validates provenance and
raw timing samples, preserves all 84 cells, and ranks each shape separately by mode.
Full-model replacement prevalidates all 36 layers before mutation; synthetic CPU tests
exercise 252 replacements and failure-policy restoration. 85 local tests: 83 pass,
2 CUDA skips. No new server/GPU evidence. See the existing
[M1 runbook](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md#本地离线汇总与流程检查).

2026-09-14 local preparation: 6 fixed M=1 configurations (baseline + 5 candidates),
12 eager/Graph trials, explicit candidate-aware block and full Qwen3-8B cached-decode
runners are prepared. No new CUDA/full-model performance result; vLLM remains
unconnected. Commands, gates and limits: [M1_CANDIDATES_FULL_MODEL_RUNBOOK.md](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md).

## Current status

Local optimization (2026-09-14): user requests local changes and explanation of
v1 slowness. Added explicit v2 candidate (bank-swizzled shared memory, sixteen
rows/CTA), preserving the v1 CUDA source and all representation/rounding gates.
Runner can compare dense/v1/v2 within one trial and requires exact v1/v2 output
agreement. No server connection or GPU validation this turn. See
[M1_V2_LOCAL_OPTIMIZATION.md](M1_V2_LOCAL_OPTIMIZATION.md). Performance is unknown;
next server run needs a fresh source-bound environment record, not the old hash set.


Latest direction (2026-09-14): user plans to close compute and iterate locally.
Do not launch new GPU work. Shutdown is planned, not yet user-confirmed.
Both M=1 trial jobs previously completed and their small evidence backups passed
hash checks. Preserve the original network volume, weights, results and caches.
Startup-time reduction plan is in [SERVER_ITERATION_PLAN.md](SERVER_ITERATION_PLAN.md):
prepare/push source before opening GPU, freeze a known image, reuse compatible
build caches, and run bounded candidate batches per startup. Current observed
install time was 39s; post-install record/compile/smoke window 62s; each real
Linear trial took about 5s. Image identity remains unresolved. That planning step did not build an image, launch a new experiment or shut down
the instance. Subsequent local candidate/full-model preparation is recorded above.


Latest M=1 trial (2026-09-14): fixed step400, layer 0 seven real Linears tested
on A100 PCIe in BF16 eager and CUDA Graph modes. All recorded numerical gates
passed. Eager k_proj dense timing was unstable (31.29% range); Graph all seven
cells passed stability. Graph packed v1 takes 2.03–2.26x dense BF16 time, so
this is accepted execution with negative speed, not an acceleration success.
See [QWEN3_8B_M1_LINEAR_RESULTS.md](QWEN3_8B_M1_LINEAR_RESULTS.md).
Both jobs exited 0; results/acceptance live under `/workspace/results/m1-a100-20260914/`.
The next action is bounded M=1 profiling/optimization, retaining v1 as baseline;
no block/full-model/vLLM job was launched. No kernel bottleneck attribution yet.


Server setup accepted (2026-09-14): RunPod A100 80GB PCIe (SM80), driver
595.91.07, Python 3.12.3, unchanged template torch 2.8.0+cu128 / CUDA 12.8.
Task `m1-env-20260914` finished with exit 0. First CUDA compile plus six smoke
tests passed in 57.79 seconds; the full server suite passed all 75 tests with
no skips. Live source hashes, clean revision `2868d5fa9763bfaa90ea1db786dea8a4c923783e`,
compiler flags (SM80, --fmad=false, no fast-math), extension binary hash and
pip check were verified. No GPU compute processes remained at review.
Status: `accepted_environment_and_synthetic_cuda`. See
[RUNPOD_M1_SETUP_RESULTS.md](RUNPOD_M1_SETUP_RESULTS.md).

The original network volume and all 43 fixed step400 manifest files passed the
startup check; snapshot directory exists but shard hashes were not rechecked.
Before/after environment, runtime activation and acceptance are under
`/workspace/results/m1-a100-20260914/`; task logs remain in
`/workspace/jobs/m1-a100-20260914-setup-v1/`.
Source `/workspace/results/m1-a100-20260914/runtime.sh` before a trial; CUDA PATH
and cache variables were task-local, not globally installed into login shells.
The subsequent real-weight M=1 trials are recorded above. The environment review
itself did not launch benchmarks; block/full-model/vLLM remain unlaunched. Base image identity/digest remains unresolved, so no
reusable image is accepted. Private verified backup:
`server_results/runpod_m1_a100_2026-09-14/`.

Latest local preparation (2026-09-14): user requests Marlin-referenced M=1 first,
then one transformer block, then full-model expansion. Future vLLM integration
is reserved at an engine-neutral interface only; do not integrate vLLM yet.
Prepared versioned lossless conversion, an uncompiled CUDA SIMT M=1 prototype,
Linear trial runner, empty-cache single-block probe and full-model replacement
adapter. See [M1_ACCELERATION_PREPARATION.md](M1_ACCELERATION_PREPARATION.md).
Local suite: 75 tests, 74 passed and one CUDA-only test skipped; compileall and
whitespace checks passed. No server connection, CUDA execution, image build or
performance claim this turn.
On the FIRST new server startup, record environment before/after dependency
setup and run build smoke; only then generate an image bundle from observed
requirements. Later recreate and revalidate an immutable image to reduce setup.
The old endpoint remains closed; preserve the network volume and frozen step400.


Latest closeout (2026-09-13): the user confirmed the compute server is closed.
See [RESULTS_OVERVIEW.md](RESULTS_OVERVIEW.md) for the consolidated results.
Fixed distilled step400 WT2 test PPL is accepted for execution: **13.169788495**,
vs its undistilled parent **14.951611048** (-11.9173%), same-run BF16 9.724944981.
Accuracy work is paused by explicit user direction; acceleration research is
future work, not currently launched. The original quality gate remains unmet.
Persistent artifact: `/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`,
36 copied layers, hashes checked, about 2.54 GiB plus small records. Manifest SHA256:
`253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`.
The `/workspace` network volume is `/networkvolumes/34au39ljvf`; preserve this
volume and the original pinned model snapshot needed for non-quantized weights.
The last pre-shutdown check found no NVIDIA compute processes, PPL exit 0, and clean
server checkout. Local closeout archive was verified. No server shutdown/delete
operation was performed by the agent; closure is user-confirmed, not a new
remote inspection. See `SERVER_SHUTDOWN_READY.md`.

Next: prepare versioned packed conversion and numerical correctness, then measure
Linear -> block -> full-model performance under `ACCELERATION_HANDOFF.md` and
`EXPERIMENT_PLAN.md`. Further accuracy optimization is not a prerequisite for
this authorized research. No acceleration implementation or GPU benchmark was
completed during the closeout. Do not reconnect to the old SSH endpoint.


## Historical 8B execution chronology

The following records preserve decisions and status at each stage. Earlier
quality-first sequencing, pending work and server state are superseded by the
current status above; completed evidence and audit caveats remain applicable.

The active research direction changed on 2026-09-12. Qwen3-8B is now the
full-model target. Work proceeds in two strictly ordered stages: first reproduce
the calibrated algorithm on 8B, produce complete quantized weights, and pass a
matched model-quality gate; only then implement and benchmark real packed
deployment. The canonical staged contract is
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

As of 2026-09-13, the pinned Qwen3-8B snapshot and new C4 calibration artifact
have passed preflight. All five layer-0 representative Linears passed independent
review on RunPod A100-SXM4-80GB; see [QWEN3_8B_LINEAR_RESULTS.md](QWEN3_8B_LINEAR_RESULTS.md).
All 45 tests passed locally and on the server before submission. Full-model
pure/hybrid artifacts from clean revision
`d8a2eff231dee7f0f4c4822179685672d60e10f9` are now **accepted_with_notes**
for integrity and weight reconstruction; see [QWEN3_8B_FULL_RESULTS.md](QWEN3_8B_FULL_RESULTS.md).
Both arms contain 36 layers / 252 Linears. Independent CPU replay covered all
504 targets; maximum relative SSE discrepancy was 3.2544e-16. Hybrid aggregate
weight SSE is 12.4301% lower; 249/252 targets improve, with three small o_proj
regressions recorded. Matched PPL execution is now accepted, but both quantized
arms fail the quality gate: BF16 **9.724945**, pure **1149.470625**, hybrid
**16.142104**. Hybrid remains **65.9866% above matched BF16**; deployment is
blocked under the frozen >10% rule. See `QWEN3_8B_PPL_RESULTS.md` for review
checks, source hashes and limitations. Further work requires an algorithm-quality
diagnosis/improvement plan; no backend or distillation was launched.

Offline diagnosis now confirms a sequential OBQ conditioning defect shared by
8B/32B: group updates reuse slices of the original inverse Hessian without
conditioning on fixed groups. A CPU quadratic counterexample fails the optimum
invariant. Its full-model PPL impact is not measured. A separate corrected hybrid
implementation and bounded probe are now prepared; the historical code remains
unchanged. See `QWEN3_8B_PPL_DIAGNOSIS.md`; preserve old artifacts and isolate this
repair before varying calibration or within-layer ordering.
User direction is now hybrid-only diagnosis, with distillation deferred. A
synthetic hybrid probe confirms that inverse conditioning can alter selected
columns, but selection policy changes must be isolated from compensation fixes.
Full metadata also shows 5329/10368 global hybrid fits hit the 50-iteration cap;
its PPL impact remains unmeasured. See the hybrid-focused follow-up in the diagnosis.

The new RunPod uses NVIDIA A100 80GB PCIe (SM80), driver 570.133.20,
Python 3.12.3 / torch 2.8.0+cu128. Retained model/calibration/artifact storage is
present. Container-disk `/opt/fluxbin-venv` was rebuilt with system-site-packages
and the existing add-on lock; pip check passed, size ~410 MB. No torch/model download.

The hybrid-only compensation probe is now accepted as a bounded Linear
**diagnostic**, after an evaluation precision repair. First run
`hybrid-compensation-probe-pcie-v1` (ae77820) failed metric agreement: the
layer-6 down corrected direct/Hessian loss differed by 0.0244%, exceeding 0.01%.
Rerun `hybrid-compensation-probe-pcie-fp64-v1`, clean revision
`3a75d7ece7e6e442471cdad74517e614ae95f3a0`, exited 0. All six comparisons pass
with maximum relative discrepancy 5.7751e-9, under the unchanged 1e-4 tolerance.
The dominant discrepancy came from FP32 Hessian accumulation: simply evaluating
the same FP32 Hessian in FP64 still gave 18.434305 versus direct 18.429785;
independent FP64 accumulation and quadratic evaluation gives 18.429785329.

Fitting still uses the original FP32 Hessian. Across both runs, input traces,
fit Hessian/target weight hashes, six payloads, selected indices and direct
losses are exactly unchanged. Direct loss reductions are 60.63%, 56.52%, 58.98%
for layer-1 gate/up and layer-6 down. This supports the compensation repair on
these targets, not a measured full-model PPL improvement. All 57 local/server
tests pass. Small evidence and independent acceptance are archived locally in
`server_results/runpod_hybrid_pcie_2026-09-13/fp64/`; old failure is preserved.
See `QWEN3_8B_HYBRID_PROBE_GUIDE.md`. The user then authorized continuing:
conditioned hybrid full-model quantization has now passed artifact/reconstruction
acceptance on the PCIe server. All 36 layers / 252 Linears / 6,945,767,424
parameters completed, exit 0, 1960.51 seconds, clean execution source
`e5f3861c22cd99bbba5cf4bb7bfdf7df3b183de5`. Independent CPU audit checked all
1,764 payload tensors and 252 reconstructions; maximum relative SSE discrepancy
3.2879e-16. Every saved selected-index hash matches both legacy and conditioned
metadata. Logs are intact. Weight SSE is 693294.065948 (8.76% above old hybrid),
which does not establish PPL quality. Full-model Hessians were not independently
replayed. Matched BF16/conditioned hybrid PPL is now accepted for execution,
but fails quality: BF16 9.724944980689296 (exact historical reproduction),
conditioned hybrid 14.951611048119167, +53.7449% versus BF16. This is a 7.3751%
PPL reduction from historical hybrid 16.142104442831975; the old hybrid was not
remeasured in this run. Source `3a3ef5c2aeee260b3b7522171b37536b7fb4d3f5`,
job `qwen3-8b-conditioned-ppl-v1`, exit 0, elapsed 109.78 seconds.
Both arms scored 146 blocks / 298862 transitions with finite metrics. Audit
rehashed snapshot/payloads/protocol/source and recomputed NLL/PPL arithmetic;
forward/logits were not independently replayed. No distillation/backend launch.
See `QWEN3_8B_CONDITIONED_PPL_RESULTS.md`; small evidence is backed up under
`server_results/runpod_hybrid_pcie_2026-09-13/conditioned-ppl/`.
See `QWEN3_8B_CONDITIONED_FULL_RESULTS.md` for hashes and evidence, and
`QWEN3_8B_CONDITIONED_FULL_GUIDE.md` for paths. Small evidence is backed up under
`server_results/runpod_hybrid_pcie_2026-09-13/full-conditioned/`.

QBB-New style scale-only distillation is now integrated: hard next-token CE +
mean post-block feature MSE, each divided by its fixed initial training value.
The complete sample200 generation/filter/validation/train/export runner passed
65 local/server tests. GPU smoke passed on A100 PCIe: all 252 initial BF16
reconstructions exact, all 1008 scales updated, frozen signs/indices unchanged,
no teacher gradients, peak allocated 26.28 GiB across tested shapes.
Formal task `qwen3-8b-distill-train-v1` completed, exit 0, source
`8b4aa4ac012d06d900bf34ee00701e1a052b0889`; acceptance is
**accepted_validation_only**. All 400 steps, 21 synthetic monitors, 7 real
validation monitors, schedules/data isolation/source hashes and final optimizer
state passed audit. All 36 exported layers/252 Linears decode finite; all 1008
scale tensors changed, fixed parent signs/indices are exactly unchanged.
Synthetic validation normalized loss 1.621944 -> 0.630516; WT2 validation PPL
15.272183 -> 13.670310 (-10.49%), teacher 10.240814425. Final step400 retained;
unsaved step300 is slightly better (13.664953), not selected post hoc.
No test evaluation during training. Post-distillation test PPL subsequently completed (13.169788495);
do not compare validation 13.670310 directly with parent's test 14.951611.
Recorded post-load elapsed 66.20 min, peak allocated 26.19 GiB. See
`QWEN3_8B_DISTILLATION_RESULTS.md`. Artifacts remain under the job directory;
small evidence archived locally in `server_results/runpod_hybrid_pcie_2026-09-13/distillation/`.

Review caveat: importing an audit from the job directory accidentally loaded
job-local `queue.py` in place of the standard library. A duplicate pure launch
was refused by existing-output protection; payloads/results were unchanged,
but the pure log/PID/exit code and queue status now describe that failed retry.
Use `results/qwen3-8b-full-hessian-obq-s8-v1/acceptance.json` as the authoritative
review record, not the overwritten queue status. Audit scripts must use the
isolated `runpy` entry from the repository root. The original pure log is lost;
this limitation and the provisional audit's overly strict all-tensor improvement
assertion are preserved in the review record. Full quantization and its audit
have finished; the separate PPL job also completed and has been reviewed.

- Server checkout: `/workspace/repos/marlin-style-fluxbin`.
- Original tmux session: `qwen3-8b-full-v1` (finished).
- Job state/logs: `/workspace/jobs/qwen3-8b-full-v1/` (`status.json`,
  `pure.log`, `hybrid_s8.log`, per-arm PID/exit-code and source manifests).
- Resolved input config/suite: checkout `results/qwen3-8b-linear-v1/full.config.json`
  and `linear-suite.json`.
- Full-model result directory: checkout `results/qwen3-8b-full-hessian-obq-s8-v1/`.
- Full-model payloads: checkout `artifacts/qwen3-8b-full-hessian-obq-s8-v1/`,
  separate `pure` and `hybrid_s8` directories.
- Full quantization provenance remains pinned to `d8a2eff`; the clean remote
  checkout was subsequently fast-forwarded to PPL revision
  `e45462aa155aeeedf72f98080a03f74852bc9120`, which remains the PPL provenance revision.
- Private local archive: `server_results/runpod_qwen3_8b_2026-09-13/`;
  full acceptance metadata, audit source and portable manifest retained, no weights downloaded.

See [QWEN3_8B_LINEAR_GUIDE.md](QWEN3_8B_LINEAR_GUIDE.md) for the input/algorithm
contract. The 8B PPL port and 48 local/server tests are complete. PPL ran in
tmux `qwen3-8b-ppl-v1`; job records live under `/workspace/jobs/qwen3-8b-ppl-v1/`
and the output is checkout `results/qwen3-8b-ppl-v1/result.json`.
Separate `acceptance.json` records `accepted_execution_quality_failed`. The
review rehashed source, model and payload files and checked protocol/counts and
NLL/PPL arithmetic; it did not independently replay forward passes or logits.
No backend/distillation auto-launch.

The user changed the environment route: use an existing RunPod PyTorch/CUDA
template first, inspect its installed packages, and add only necessary missing
dependencies. A custom image is no longer a prerequisite for initial checks.
The user provisioned the A100 Pod; the environment is installed on container
disk at `/opt/fluxbin-venv`, while model/data/artifacts stay on `/workspace`.

RunPod reconstruction is addressed by the versioned bootstrap image and
persistent-storage contract in [`infra/runpod/README.md`](../infra/runpod/README.md).
The intended lifecycle is terminate compute, retain the Network Volume, and
recreate a Pod from the same Template and immutable image.

Custom-image provisioning remains deferred. GitHub Actions run
[`34681093330`](https://github.com/maxiu188-afk/Marlin_Style_Fluxbin/actions/runs/34681093330)
at commit `e041f77c8edf5c2ce095c18ec003e66582039e83` failed inside the
`Build and push linux/amd64 image` step because the hosted runner reported
`No space left on device`. The digest-recording step never ran, so no image is
accepted or referenced by digest. That workflow created no RunPod resources;
the current A100 Pod was provisioned separately by the user. The workflow now
requires manual dispatch and must remain idle
until a build-space strategy is approved.

The assignment-overhead repair, complete Qwen3-32B v3 full-model
reconstruction, exact replay gate, and matched WikiText-2 PPL execution remain
accepted historical evidence. Hybrid-s8 is materially better than pure
two-base OBQ, but it remains above BF16 quality. Qwen3-32B is no longer an
active full-model deployment target; retain its exact Linear shapes only for
operator-level scaling tests.

Historical accepted Qwen3-32B checkpoints:

- full-model source revision:
  `412764e04cc5c997e7bc135ed52b128d08897a61`;
- PPL source revision: `e6921426e8e2249268f0861505556c12adc603e0`;
- model: `Qwen/Qwen3-32B` at
  `9216db5781bf21249d130ec9da846c4624c16137`;
- full-model config:
  `configs/experiments/qwen3_32b_full_hessian_obq_s8_v3.json`;
- PPL config:
  `configs/evaluation/qwen3_32b_wikitext2_full_hessian_obq_s8_v3.json`;
- artifact id:
  `qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized`.

No backend or distillation job was launched from the historical 32B result.

## Original Qwen3-8B plan (sequencing superseded)

- Target `Qwen/Qwen3-8B` at previously accepted revision
  `b968826d9c46dd6066d109eabc6255188de91218`, subject to an exact preflight.
- Expected scope: 36 layers, 252 transformer-block Linears, and
  6,945,767,424 included weight elements.
- Preserve the current two-base Hessian-OBQ and hybrid-s8 algorithm contract so
  model size is the only planned algorithm-stage change.
- First gate representative real Linears; then produce complete pure/hybrid
  artifacts; then run matched BF16/pure/hybrid WikiText-2 PPL.
- Only an arm within 5% relative PPL of matched BF16 may enter deployment. A
  gap above 10% blocks deployment; the intermediate range requires a separately
  approved quality-improvement plan.
- After quality acceptance: define a versioned packed runtime layout, pass
  conversion and CUDA correctness, measure real-shape operators, integrate a
  block, and finally measure the complete model and serving path.
- H20 is the iterative SM90 development platform. H200 and A100 80GB provide
  the formal final Hopper and SM80 evaluations.
- Qwen3-32B shapes remain operator-only stress cases. No new 32B full-model
  quantization, PPL, or serving work is authorized.

## Historical Qwen3-32B calibrated algorithm contract

- C4 calibration uses 256 sequences. Sequence length 2048 and seed `20260902`
  are explicit project choices because the paper does not specify them.
- Accumulate the scalar-normalized equivalent of `H = 2 X^T X`; apply 1%
  mean-diagonal damping before Cholesky inversion. The damping value is also a
  project-frozen choice.
- Use exactly two binary bases, group size 128, independent row/column scales,
  exact four-pattern sign assignment, the existing greedy initializer, and a
  50-step ALS limit.
- Pure two-base is a complete independent OBQ arm with no residual refinement.
- Hybrid-s8 chooses 8 group-local columns before global decomposition using
  `sum_i(W_ij^2) / Hinv_jj^2`, with stable lower-index tie breaking, and fits
  sparse residual refinement.
- After each group, each arm updates all unprocessed columns using its own group
  error and inverse-Hessian blocks.
- Each branch propagates its quantized hidden states into the next layer. Pure
  and hybrid therefore do not share global payloads or later-layer Hessians.
- No Shared-C, distillation, packed CUDA kernel, or serving integration.

## Accepted assignment-overhead repair

The old fixed 16-row assignment chunk caused about 10.24 million synchronizing
`.item()` calls per pure layer and 17.29 million per hybrid layer. Revision
`eef867dfa37ad2b5e2cd848eb4330bd99c46d313` introduced a 1 GiB adaptive
temporary-memory budget and accumulated assignment-change counts on device.
The active group-local OBQ fit now processes all 5120 output rows in one chunk,
while the legacy full-weight path remains memory-bounded.

Single-Linear regression job `6282732` completed `0:0` on Isambard GH200 in 51
seconds and reproduced accepted job `6271396` bit-for-bit:

- all 10 payload tensors and solver records matched;
- payload SHA-256 remained
  `bcddf5b77fb679f6784129c20bcd54e734a40369cccf78fbff5bc370d369583f`;
- application time: `571.1411166 -> 27.7813934` seconds (`20.5584x`);
- Slurm wall time: `00:09:45 -> 00:00:51` (`11.4706x`);
- peak allocated memory remained exactly `71,720,856,576` bytes.

This accepts the semantics-preserving assignment repair. It does not claim a
packed-kernel speedup.

## Complete v3 full-model execution

### Bounded replay and exact gate

The first v3 jobs regenerated exactly the previously accepted layer scope using
the repaired implementation:

| Job | Arm / scope | State | Slurm time | Application time |
| --- | --- | --- | ---: | ---: |
| `6283506` | pure, layers 0-10 | `COMPLETED 0:0` | `00:11:57` | `533.2232 s` |
| `6283507` | hybrid-s8, layers 0-5 | `COMPLETED 0:0` | `00:10:12` | `417.0263 s` |
| `6283508` | exact comparison gate | `COMPLETED 0:0` | `00:00:20` | - |

The repaired averages were `48.4748` seconds per pure layer and `69.5044`
seconds per hybrid layer. Against the matched pre-repair means of `2593.999`
and `4662.264` seconds per layer, this is approximately `53.51x` and `67.08x`.

Gate `6283508` passed both arms:

- pure layers 0-10: 11 layers, 231 payload tensors, 1,681,334,512 bytes;
  all payloads bit-exact and all algorithm metadata exact; comparison SHA-256
  `8297431e588dca751db37f224c723dd3d1cebd8576a97244239b2c51ba324f68`;
- hybrid layers 0-5: 6 layers, 294 payload tensors, 1,145,889,312 bytes;
  all payloads bit-exact and all algorithm metadata exact; comparison SHA-256
  `e6c8db53a459b896f431fb5b8baf3b55fd64da8e3575d6f314b1517aee58c686`;
- gate result SHA-256:
  `b3a97237f5352ecbdf786cf0f90230b4d470c3bc2c5c5357f3556e7ee723fe25`;
- `continuation_authorized=true`; neither PPL nor backend auto-launched.

Two earlier boundary-test jobs were stopped and quarantined after exposing an
inclusive layer-boundary bug. Revision
`412764e04cc5c997e7bc135ed52b128d08897a61` fixed the boundary before any
accepted v3 replay or continuation.

### Resume and final inventory

| Job | Arm | Replayed layers | New layers | State | Slurm time |
| --- | --- | ---: | ---: | --- | ---: |
| `6283509` | pure | 11 | 53 | `COMPLETED 0:0` | `00:45:14` |
| `6283510` | hybrid-s8 | 6 | 58 | `COMPLETED 0:0` | `01:06:45` |

Both final arms cover 64 layers, 448 Linears, and 31,205,621,760 weights.
Independent server-side acceptance rehashed all 128 metadata/payload pairs
(about 22.01 GB), validated every Safetensors inventory and internal tensor
hash, confirmed contiguous layers, and found finite metrics throughout.

| Metric | Pure | Hybrid-s8 | Hybrid change vs pure |
| --- | ---: | ---: | ---: |
| Weight SSE | `2,337,870.7642333917` | `2,117,499.284547236` | `-9.42616175%` |
| Relative Frobenius | `0.37708974135338075` | `0.35887739498900334` | `-4.82971144%` |
| Branch-calibrated output SSE | `1,155,273.3216937627` | `394,157.5088691871` | `-65.88188254%` |

The output-SSE comparison is branch-specific: later layers have different
propagated inputs and Hessians. It is not a same-Hessian comparison.

Final result SHA-256 values:

- pure: `4b9f5ab5c41bf7e266fc8a4c52072db9fb7869437bb871d3c55c14d243078e08`;
- hybrid-s8:
  `9a8753cdb99efc224651c5ca270efe050e44c58d4c1ccd66133c5d132924746a`.

The JSON status remains `completed_pending_review` because the producer is
fail-closed. Manual review is now complete and accepts the execution and
artifacts.

## Accepted WikiText-2 PPL gate

Job `6296175` ran on GH200 node `nid010773` from clean revision
`e6921426e8e2249268f0861505556c12adc603e0`; Slurm recorded `COMPLETED 0:0` in
`00:06:04`.

The run reused the accepted QBB-New WikiText-2 protocol without resampling:

- 146 non-overlapping 2048-token blocks;
- 299,078 tokens and 298,862 scored transitions;
- token SHA-256
  `c7a8c41e587561b93c8dd0b17224e6f20aa4270c9dab62151357cca88651ad9e`;
- BF16 reference job `6154681`, reproduced with absolute difference `0.0`.

| Arm | Mean NLL | PPL | Relative to BF16 |
| --- | ---: | ---: | ---: |
| BF16 | `2.0295734206653275` | `7.6108390395557874` | `1.000000x` |
| Pure | `2.876087261093014` | `17.744706766201215` | `2.33150467x` |
| Hybrid-s8 | `2.3450575398268865` | `10.433873073527254` | `1.37092284x` |

Hybrid-s8 reduces PPL by `7.3108336927` (`41.20008174%`) versus pure. Pure is
`133.150467%` above BF16 and is rejected as a final candidate. Hybrid-s8 is the
current best calibrated arm, but remains `37.092284%` above BF16; it is not
accepted as BF16-equivalent or production-ready.

All three arms scored exactly 298,862 transitions with no non-finite blocks.
For each quantized arm, materialization covered 64 layers, 448 Linears, and
31,205,621,760 parameters, and validated all payload and internal tensor hashes.
The application result elapsed time was `178.0604` seconds; the larger Slurm
time includes preflight tests, manifests, and hashing.

- PPL result SHA-256:
  `83b682aae91bbbde1b39b1b31c56716f2497501d220751b322ce6fb4dd861325`;
- PPL source-manifest SHA-256:
  `9644187b14a3e04462092bd6d5e9c2799d339975daa401ed2e91e77ff601e4eb`.

This is dense fake-quant PPL. It does not establish packed representation
correctness, kernel performance, end-to-end latency, throughput, or serving.

## Retention and private export

On 2026-09-13 the user authorized removal of downloaded local 32B quantized
weights. Fourteen hybrid payload files (including one incomplete download),
2,505,730,544 logical bytes, were removed; all 14 layer metadata files, result
JSON, logs and calibration evidence were retained. Remote state was not changed
or reverified. Private provenance records this in
`server_results/provenance/local_32b_payload_removal_2026-09-13.json`.
Older statements below about weights not downloaded describe the earlier export.

The obsolete partial full-model v2 artifact directory, result directory, and
jobs `6272553`/`6272554` logs were removed from Isambard. The corresponding
partial full-v2 data was also removed from local `server_results/`. The accepted
C4 calibration, WikiText-2 protocol/BF16 reference, and single-Linear oracle
remain because v3 provenance depends on them.

The private, Git-ignored `server_results/` bundle contains:

- v3 bounded-replay and final full-model result JSON/source manifests;
- the exact replay-gate JSON/source manifest;
- the v3 PPL result/source manifest;
- the associated full-model, gate, PPL, calibration, and single-Linear logs;
- calibration/protocol dependencies and transfer provenance.

The v3 full-model weights were deliberately not downloaded. Remote-to-local
checksum dry runs found no file-content differences. The current local
`provenance/SHA256SUMS` SHA-256 is
`26d72ab7d660b13d4334acd2ef4cbe47261180146595746e1db9199cb30fe70e`.
Nothing under `server_results/` belongs in Git.

## Implementation map

- `src/fluxbin_style/hessian_obq.py`: Hessian accumulation/inversion, saliency,
  OBQ propagation, and independent pure/hybrid groupwise quantization.
- `src/fluxbin_style/two_base_rank1.py`: two-base rank-one fit and adaptive
  assignment chunking.
- `src/fluxbin_style/qwen3_sequential.py`: Qwen3 layer capture, materialization,
  propagation, and resume support.
- `scripts/run_qwen3_full_hessian_obq_s8.py`: branch-specific full-model runner.
- `scripts/compare_full_hessian_obq_artifacts.py`: exact pre/post-repair gate.
- `scripts/run_qwen3_full_hessian_obq_s8_ppl.py`: v3 dense fake-quant PPL gate.
- `scripts/run_isambard_qwen3_32b_full_hessian_obq_s8.sbatch`: full-model Slurm
  entrypoint.
- `scripts/run_isambard_qwen3_32b_full_hessian_obq_s8_ppl.sbatch`: PPL Slurm
  entrypoint.

## Historical 32B evidence boundary and original next action

The algorithm-quality ladder through dense fake-quant PPL is complete. The
accepted result says:

- the runtime repair preserved the algorithm exactly;
- both 64-layer branches are complete and internally valid;
- hybrid-s8 materially improves pure in reconstruction and PPL;
- hybrid-s8 still has a substantial gap to BF16.

There was no predeclared deployment-quality threshold for the historical 32B
run, so its PPL execution remains accepted without claiming production quality.
It does not authorize a packed backend.

The next authorized work is planning and implementation for Qwen3-8B Phase A0:
create versioned 8B configs/runners, verify the pinned model inventory and
calibration protocol, and run local static/synthetic checks. Real-model work
must then follow the gates in `EXPERIMENT_PLAN.md`; no phase auto-launches the
next one.

## Environment boundary

- macOS/Apple Silicon: source review, static compilation, and small local tests.
- Isambard GH200: historical accepted 32B real-weight, exact-gate, full-model,
  and PPL evidence only; it is not the assumed environment for the new plan.
- H20: planned Qwen3-8B SM90 development and tuning.
- H200 and A100 80GB: planned formal final performance evaluation.
- Scheduler `COMPLETED` alone is not acceptance. Structured JSON, provenance,
  hashes, inventory, finite metrics, and the declared gate must all pass.

## Historical note

The earlier v1 weight-only full-model result and PPL execution were valid but
negative: pure PPL `147.5315`, hybrid-s8 PPL `24.2153`, versus BF16 `7.61084`.
They are historical algorithm evidence only. The obsolete partial calibrated
full-model v2 run was superseded by the exact-gated complete v3 run and is no
longer retained.
