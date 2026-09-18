# Qwen3-8B offline-rotated hierarchical W2 runbook

A single-arm diagnostic: fold an offline-fusible QuaRot-style rotation into the
BF16 checkpoint, run the unchanged hierarchical-W2 GPTQ pipeline at H2.50, and
score it against a same-run W3 replay. It answers one question — does rotation
move the 2.500 bpw arm enough to be worth continuing — and stops.

It is a separate experiment with its own config and its own source-hash set.
The accepted unrotated H2.50/H2.875 artifacts are untouched: no file in the
unrotated runner's `SOURCE_FILES` changes, so their `implementation_sha256`
still validates.

## What the rotation does and does not cover

`src/fluxbin_style/offline_rotation.py` applies only the two rotations that
fold into ordinary parameters:

| Rotation | Absorbed into | Covered |
|---|---|---|
| `R1` residual, 4096, Walsh | embedding, q/k/v/gate/up inputs, o/down outputs, `lm_head` | yes |
| `R2` per-head, 128, Walsh | `v_proj` outputs, `o_proj` inputs | yes |
| `R3` QK online Hadamard | — | **no** |
| `R4` MLP online Hadamard | — | **no** |

`R4` would rotate the activation entering `down_proj`, which is an elementwise
SwiGLU output and cannot be folded into any weight matrix. So `down_proj` keeps
its original 12288 input basis — the one Linear with the largest `in_features`,
and 26% of each layer's weights, gets no input-side incoherence benefit. Expect
the result to be a lower bound on what full QuaRot would give.

Qwen3's per-head `q_norm`/`k_norm` sit after `q_proj`/`k_proj`, so `R1` (which
only changes those projections' input basis) leaves them bit-identical. This is
asserted in `tests/test_offline_rotation.py`.

The transform is exact in real arithmetic: with the RMSNorm scale fused into
the following Linear, a unit RMSNorm commutes with an orthogonal rotation. The
local equivalence test measures `1.08e-06` relative logit deviation in FP32,
which is round-off, with identical greedy tokens.

## Prior evidence, and how to read it

From the QuaRot/SpinQuant reproduction workspace on Llama-2-13B:

| Setting | Unrotated | Rotated | Verdict |
|---|---:|---:|---|
| **W4A16, weight-only** | 5.289677 | **5.132755** | rotation recovered 55.67% of the loss |
| W4A8, INT8 activations | 5.148000 | 5.162700 | rotation slightly worse |
| W4AFP8, FP8 activations | 5.136105 | 5.248356 | rotation worse |

Every negative row involves activation quantization. This experiment is
weight-only with a dense BF16 evaluation, so the W4A16 row is the structurally
matched one. Confirm that the W4A16 unrotated arm also used `actorder`, since
this pipeline runs `desc_act=True`; if it did not, discount the expected gain.

## Decision boundary

`report_only_flags.below_continuation_threshold` reports whether the rotated
H2.50 lands under 20.0 PPL against the frozen unrotated 27.932835. The flag is
report-only and the continuation decision is manual.

A rotated result at or above roughly 20 is a manual stop signal for this
bounded offline-only route. It does not identify whether the remaining error
comes from scale fitting, the W2 codebook, layer sensitivity, the fixed
rotation, or an excluded online transform. Do not resume H2.625/H2.75, add
arms, launch learned rotations (SpinQuant), distillation, mixed precision, or
kernel work off the back of this number.

The 20.0 threshold comes from a single-point extrapolation of the proxy-loss to
PPL exponent (~3.3) measured against this one unrotated pair. It is a
preregistered line to keep the decision honest, not a prediction.

## Required graded probe before the full model

Do not start the 36-layer GPTQ/PPL job directly. The probe wrapper runs two
paired H2.50 stages and stops automatically when a gate fails:

| Stage | Layers | Calibration | Pass gate | Next action |
|---|---|---:|---|---|
| `layer0` | 0 | first 32 frozen C4 sequences | block-error ratio <= 0.95, layer win, >=4/7 Linear wins, down ratio <=1.25 | run representative probe |
| `representative` | add 17 and 35; reuse the bound layer-0 result | same first 32 sequences | combined 0/17/35 aggregate ratio <=0.85, >=2/3 layer wins, no layer >1.05, >=14/21 Linear wins, down ratio <=1.05 | eligible for manual full-model launch |

The primary metric is DecoderLayer output squared error normalized by that
arm's pre-quantization output energy. The local pre-quantization output is an
internal error reference only; the probe does **not** add a rotated-BF16 PPL
arm. Upstream layers stay in that arm's unquantized BF16 parameterization, so
this is a local layer-quality test rather than a sequential error-propagation
or PPL result. It stores JSON metrics only, not quantized payloads or
checkpoints.

Across both gates, only 42 Linear projections are computed: 14 at layer 0
(seven Linears x two arms), then 28 at layers 17/35. This is one sixth of the
252 projections in one full-model arm, while Hessian capture uses 32/256 of
the frozen calibration sequences. Model loading, offline rotation, and BF16
teacher traversal remain fixed overhead and are reported separately.

Launch the graded probe on the pinned RTX PRO 4500 route:

```bash
cd /workspace/repos/marlin-style-fluxbin
source /workspace/runtime/current/runtime.sh

job=/workspace/jobs/hierarchical-w2-rotation-probe-$(date -u +%Y%m%dT%H%M%SZ)
output=/workspace/results/qwen3-8b-hierarchical-w2-rotation-probe-v1

tmux new-session -d -s hierarchical-w2-rotation-probe \
  "FLUXBIN_PYTHON=$(command -v python) FLUXBIN_EXECUTION_POLICY=same-device-quality bash scripts/run_qwen3_8b_hierarchical_w2_rotation_probe_job.sh /workspace '$job' '$output'"
```

Check only the bounded state and latest log:

```bash
cat "$job/status"
tail -n 40 "$job/layer0.log"
tail -n 40 "$job/representative.log" 2>/dev/null || true
```

`completed_probe_passed_pending_manual_full_model` does not launch or accept
the full model. Review both JSON files first, then make a separate manual
decision. Either `completed_probe_stopped_*` status is a valid negative result.

## Launch

Only after the representative probe passes and its JSON is reviewed, use this
formal full-model command from a clean checkout on the same device class as the
accepted endpoint run (A100 80GB PCIe):

```bash
cd /workspace/repos/marlin-style-fluxbin
source /workspace/runtime/current/runtime.sh

job=/workspace/jobs/hierarchical-w2-rotated-$(date -u +%Y%m%dT%H%M%SZ)
output=/workspace/results/qwen3-8b-hierarchical-w2-rotated-v1

tmux new-session -d -s hierarchical-w2-rotated \
  "FLUXBIN_PYTHON=$(command -v python) bash scripts/run_qwen3_8b_hierarchical_w2_rotated_job.sh /workspace '$job' '$output'"
```

For the pinned RTX PRO 4500 route, prefix `FLUXBIN_EXECUTION_POLICY=same-device-quality`;
the status then becomes `completed_cross_device_rotated_pending_review` and the
row is never an A100 reproduction. The W3 replay and the rotated arm must share
one device.

Bounded status checks:

```bash
cat "$job/status"
tail -n 40 "$job/quant-rotated.log"
tail -n 40 "$job/ppl.log"
```

The output root resumes per-layer artifacts and refuses a different config
hash. Review `$output/ppl-rotated-result.json`; its `comparison` block carries
the rotated PPL, the gain over the frozen unrotated baseline, the gap to the
same-run W3, and the recovered fraction of the unrotated-to-W3 gap.

## Copy the raw JSON back

The unrotated endpoint run left its raw artifacts only on the network volume.
Copy `$output/ppl-rotated-result.json` and `$output/H2.50-rotated/result.json`
to the Mac before releasing the pod this time.
