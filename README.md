# Marlin-Style FluxBin

Local v5 LUT-A16 preparation (2026-09-15): eight-sign FP32 lookup tables,
row-owned accumulation, fused table construction, no A8 quantization. Linear,
block and full-model entry points are connected; vLLM remains unconnected.
Local suite: 97 passed, 5 CUDA skips. NVCC, GPU correctness and performance are
pending; v4 remains the latest GPU evidence. See [v5 run commands](docs/M1_CANDIDATES_FULL_MODEL_RUNBOOK.md#v5-lut-a16-本地准备2026-09-15).

Latest v4 GPU result (2026-09-15): A100 SXM4, 98 tests passed; candidate numerical
checks 84/84, stable timing cells 80/84. Selected v4/gps4 did not beat v3 or dense.
Full-model report-only completed in 119s with both prompts stable: 0.83063x /
0.83143x original BF16 speed, no acceleration. Unstable block timing was explicitly
allowed without changing numerical/route/provenance gates. Raw evidence is backed
up and hash-verified; no experiment processes remain. Persistent venv reuse took
36.9s (30.2s imports), versus local imports 2.8–3.1s; trials used local venv.
See [v4 results](docs/QWEN3_8B_M1_LINEAR_RESULTS.md). Earlier local/pending statements are historical.

Local v4 implementation: column-scaled activations are prepared once per group,
then sign dot products are row-scaled, with eight sparse positions handled separately.
The new structural FP64 Linear reference is independent of legacy BF16 weights;
all three launches are timed. CUDA compilation/performance remain unvalidated.
See [v4 contract](docs/M1_V2_LOCAL_OPTIMIZATION.md).

Latest GPU trial (2026-09-15): v3 compiled on A100 80GB PCIe; all 91 tests passed.
The fixed candidate batch passed 84/84 numerical cells (67/84 stable timings).
Selected v3/gps4 improved Linear timing over v1/v2 but remained slower than dense.
Full-model report-only run exited 0 in 312s: prompt 0 achieved 0.86887x original BF16
speed; prompt 1 has an unstable auxiliary decoded baseline, so overall status is
`completed_unstable`. No full-model speedup. Evidence downloaded/hash-verified;
no experiment processes remain. See [v3 results](docs/QWEN3_8B_M1_LINEAR_RESULTS.md).
Earlier preparation and SXM4 statements below are historical.

Local kernel follow-up: explicit `v3` now implements three-stage cp.async staging,
register fragment double buffering and FP16/BF16 Tensor Core MMA for hybrid M=1.
It retains the v1 layout and a deterministic split reduction, with a direct-store
single-split path. This is implementation preparation only: no NVCC/GPU validation
or speed result yet. See [kernel design](docs/M1_V2_LOCAL_OPTIMIZATION.md) and the new fixed Marlin candidate config.

Latest completed full-model trial (2026-09-14): report-only run exited 0 in 135s.
On A100 SXM4, packed decode is 0.9332x / 0.9562x the original BF16 speed on the two
fixed prompts (latency +7.16% / +4.58%); all decode timings and 252-Linear route
checks passed. Numerical differences were retained under the user-authorized
report-only policy. No full-model speedup was achieved. Details: [full-model results](docs/QWEN3_8B_M1_LINEAR_RESULTS.md).

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
running, not accepted. See [SXM4 results](docs/QWEN3_8B_M1_LINEAR_RESULTS.md).
Earlier unlaunched/preparation statements below are historical.

Project documentation is collected in [docs/README.md](docs/README.md). Start with
[the current handoff](docs/CURRENT_HANDOFF.md) for progress and next steps.

2026-09-14 local preparation: 6 fixed M=1 configurations (baseline + 5 candidates),
12 eager/Graph trials, explicit candidate-aware block and full Qwen3-8B cached-decode
runners are prepared. No new CUDA/full-model performance result; vLLM remains
unconnected. Commands, gates and limits: [M1_CANDIDATES_FULL_MODEL_RUNBOOK.md](docs/M1_CANDIDATES_FULL_MODEL_RUNBOOK.md).

This repository studies a calibrated two-base rank-one binary weight representation.
The active target is Qwen3-8B; Qwen3-32B remains historical algorithm-quality
evidence and a source of operator-only stress shapes.

2026-09-14 first M=1 trial: seven real layer-0 Linears pass numerical checks.
CUDA Graph timings are stable but packed v1 takes 2.03–2.26x the same-run dense
BF16 time. This is a negative performance baseline; block/full-model/vLLM remain
unlaunched. See [M=1 Linear results](docs/QWEN3_8B_M1_LINEAR_RESULTS.md).

Historical local acceleration preparation: M=1 CUDA prototype and lossless layout,
Linear and single-block trial entries, full-model replacement adapter, and a
future vLLM interface are prepared. GPU compilation/correctness/timing remain
unverified; no server was contacted. Follow [the M=1 preparation guide](docs/M1_ACCELERATION_PREPARATION.md),
including first-server environment capture and subsequent image preparation.

2026-09-13 closeout: full-model reconstruction, compensation repair, scale-only
distillation and fixed step400 test PPL have been reviewed. Start with the
[results overview](docs/RESULTS_OVERVIEW.md) for the evidence and source revisions.

| Qwen3-8B weights | WikiText-2 test PPL |
| --- | ---: |
| BF16 | 9.724945 |
| Original pure | 1149.470625 |
| Original hybrid | 16.142104 |
| Conditioned hybrid, undistilled | 14.951611 |
| Conditioned hybrid, distilled step400 | **13.169788** |

Distillation improves test PPL by **11.92%** versus its direct parent. Separately,
WT2 validation PPL improved from 15.272183 to 13.670310. The test table combines
matched-protocol runs, not five arms measured in one run; BF16 reproduced exactly.
The final model remains 35.42% above BF16 and does not pass the original quality gate.

Accuracy experiments are paused by user direction. Research on packed conversion,
CUDA correctness and acceleration is authorized next, independently of the unmet
quality gate; no packed CUDA or end-to-end speed result exists yet. Follow
[the acceleration handoff](docs/ACCELERATION_HANDOFF.md) and
[the staged plan](docs/EXPERIMENT_PLAN.md).

The user confirmed the compute server is closed. Before shutdown, the 36-layer
step400 payload and records were copied and hash-verified on the persistent
network volume; large weights were not downloaded locally. See
[storage and restart details](docs/SERVER_SHUTDOWN_READY.md). Use an existing RunPod
PyTorch/CUDA template and restore only missing dependencies on container disk.

The reusable RunPod image, Network Volume, Template, and cache layout are
specified in [`infra/runpod/README.md`](infra/runpod/README.md).

Custom-image provisioning remains deferred. GitHub Actions run `34681093330`
failed during image build/push because the hosted runner exhausted its disk;
no custom image digest was accepted, and that workflow created no RunPod
resources. The now-closed A100 Pod was provisioned separately by the user from an
existing template. The image workflow remains manual-dispatch only.

## Historical accepted Qwen3-32B result

The assignment-overhead repair, the complete 64-layer v3 reconstruction, and
the matched WikiText-2 PPL execution have been accepted after structured-result,
provenance, hash, inventory, and finite-metric checks.

| Arm | Weight SSE | Relative Frobenius | WikiText-2 PPL | Relative to BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | - | - | `7.6108390396` | `1.0000x` |
| Pure two-base OBQ | `2,337,870.7642` | `0.3770897414` | `17.7447067662` | `2.3315x` |
| Hybrid-s8 OBQ | `2,117,499.2845` | `0.3588773950` | `10.4338730735` | `1.3709x` |

Hybrid-s8 reduces full-model weight SSE by `9.4262%` and PPL by `41.2001%`
relative to pure. It is the better of the two calibrated arms, but its PPL is
still `37.0923%` above BF16. The execution evidence is accepted; the result is
not BF16-equivalent and does not yet justify packed-backend or serving work.

The PPL path materializes the packed algorithm payload into dense BF16 weights.
It is algorithm-quality evidence, not packed-kernel correctness, latency,
throughput, or serving evidence.

## Calibrated algorithm contract

- Model: `Qwen/Qwen3-32B`, revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Calibration: 256 C4 sequences, project-frozen at 2048 tokens and seed
  `20260902`.
- Hessian: scalar-normalized equivalent of `H = 2 X^T X`, with Cholesky
  inversion and 1% mean-diagonal damping.
- Representation: exactly two `{-1,+1}` bases, group size 128, independent row
  and column scales, and exact four-pattern assignment.
- Pure is a complete independent OBQ arm with no residual refinement.
- Hybrid-s8 selects 8 columns per 128-column group using
  `sum_i(W_ij^2) / Hinv_jj^2`, then fits sparse residual refinement.
- Both arms propagate their own quantization error and hidden states; they do
  not share global payloads.
- The greedy initializer and 50-step ALS limits are unchanged.
- No Shared-C, distillation, CUDA kernel, or serving integration is included.

The paper fixes C4 and 256 calibration samples but not sequence length, seed,
or inverse-Hessian damping; those values are explicit project choices.

## Runtime repair

The original 16-row assignment loop synchronized millions of `.item()` calls
per layer. Revision `eef867dfa37ad2b5e2cd848eb4330bd99c46d313` replaced it
with memory-budgeted adaptive row chunks and one device-to-host synchronization
per assignment.

Single-Linear GH200 regression job `6282732` reproduced all 10 payload tensors
bit-for-bit against accepted job `6271396`; payload SHA-256 remained
`bcddf5b77fb679f6784129c20bcd54e734a40369cccf78fbff5bc370d369583f`.
Application time fell from `571.1411` to `27.7814` seconds (`20.5584x`) and
Slurm wall time from `00:09:45` to `00:00:51` (`11.4706x`), with unchanged peak
allocated GPU memory.

The bounded v3 replays then averaged `48.4748` seconds per pure layer and
`69.5044` seconds per hybrid layer, versus `2593.999` and `4662.264` seconds in
the matched pre-repair runs. The corresponding application-time improvements
are approximately `53.51x` and `67.08x`.

## Complete v3 full-model reconstruction

Config: `configs/experiments/qwen3_32b_full_hessian_obq_s8_v3.json`.

Artifact id: `qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized`.

Execution revision: `412764e04cc5c997e7bc135ed52b128d08897a61`

The accepted scope is all 64 transformer layers, 448 Linears, and
31,205,621,760 weights per arm. Embeddings, output head, and non-matrix tensors
remain outside the contract.

- Bounded replay jobs `6283506` (pure layers 0-10) and `6283507` (hybrid layers
  0-5) completed in `00:11:57` and `00:10:12`.
- Exact gate job `6283508` proved both payload and algorithm metadata equality
  against the retained pre-repair layers: 231 tensors / 1,681,334,512 bytes for
  pure, and 294 tensors / 1,145,889,312 bytes for hybrid.
- Resume jobs `6283509` and `6283510` completed the remaining pure and hybrid
  layers in `00:45:14` and `01:06:45`.
- The independent server audit rehashed all 128 layer metadata/payload pairs,
  checked every Safetensors inventory and internal tensor hash, and found no
  non-finite metrics.

Final result SHA-256 values:

- pure: `4b9f5ab5c41bf7e266fc8a4c52072db9fb7869437bb871d3c55c14d243078e08`
- hybrid-s8: `9a8753cdb99efc224651c5ca270efe050e44c58d4c1ccd66133c5d132924746a`
- exact replay gate: `b3a97237f5352ecbdf786cf0f90230b4d470c3bc2c5c5357f3556e7ee723fe25`

Hybrid's branch-calibrated output SSE is `394,157.5089`, `65.8819%` below
pure's `1,155,273.3217`. Because later layers use arm-specific propagated
inputs and Hessians, this is a branch-specific protocol diagnostic, not a
same-Hessian comparison.

## Accepted WikiText-2 PPL

Config: `configs/evaluation/qwen3_32b_wikitext2_full_hessian_obq_s8_v3.json`.

Execution revision: `e6921426e8e2249268f0861505556c12adc603e0`.

GH200 job: `6296175`, `COMPLETED 0:0`, `00:06:04`

The gate reused the accepted 146-block, 2048-token WikiText-2 artifact:
299,078 tokens and 298,862 scored transitions, with token SHA-256
`c7a8c41e587561b93c8dd0b17224e6f20aa4270c9dab62151357cca88651ad9e`.
All three arms scored exactly the same transitions with finite metrics and no
non-finite blocks. The accepted BF16 PPL from job `6154681` was reproduced with
zero difference.

Result SHA-256:
`83b682aae91bbbde1b39b1b31c56716f2497501d220751b322ce6fb4dd861325`.

## Evidence and retention

The private, Git-ignored `server_results/` bundle contains the accepted v3
result JSON, source manifests, exact-gate outputs, PPL output, logs, calibration
inputs, and provenance. The v3 full-model weights were deliberately not
downloaded. The obsolete partial full-model v2 artifacts/results/logs were
removed from both Isambard and the local bundle; the accepted calibration and
single-Linear oracle remain because v3 provenance depends on them.

The bundle's `provenance/SHA256SUMS` file has SHA-256
`26d72ab7d660b13d4334acd2ef4cbe47261180146595746e1db9199cb30fe70e`.
`server_results/` is private evidence and must not be committed.

## Evidence ladder

The historical Qwen3-32B ladder is complete through dense fake-quant PPL. The
new active Qwen3-8B ladder is:

1. Port/preflight and synthetic algorithm checks.
2. Representative real-Linear pure/hybrid gates.
3. Complete independently propagated 36-layer, 252-Linear quantization.
4. Matched BF16/pure/hybrid WikiText-2 PPL and the frozen quality decision.
5. Versioned deployment-layout conversion and exact correctness.
6. CUDA correctness, then real-shape operator performance.
7. Block integration and direct block timing.
8. Full-model and serving latency, throughput, memory, and correctness on H20
   during development, with final H200 and A100 80GB evaluation.

No runner automatically launches its successor.

## Local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

macOS/Apple Silicon is used for source review and local tests. The historical
accepted real-weight, full-model, and PPL evidence was produced on Isambard
GH200. New real-model and CUDA work must run on the NVIDIA environment named by
the active plan. Scheduler/process completion alone is never treated as
acceptance.

For the full operational record and exact acceptance boundaries, see
[`CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md).
