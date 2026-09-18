# Qwen3-8B hierarchical-scale W2 quality runbook

## 2026-09-17 endpoint result

The bounded endpoint run completed on one NVIDIA A100 80GB PCIe with source
revision `457008a`. The job exited 0 with status
`completed_endpoint_pending_review`; all three scored arms used the frozen
146 x 2048 WikiText-2 protocol and each scored 298,862 transitions.

| Arm | Nominal bpw | Actual tensor bpw | WikiText-2 PPL | PPL gap vs W3 |
|---|---:|---:|---:|---:|
| BF16 frozen A100 reference (not same-run) | 16.000 | 16.000000 | **9.724945** | -1.541863 |
| GPTQ W3 g128 reference | 3.125 | 3.154552 | **11.266808** | - |
| H2.50 (U4/U4) | 2.500 | 2.506115 | 27.932835 | +16.666027 (+147.92%) |
| H2.875 (U8/U8) | 2.875 | 2.881115 | 27.898930 | +16.632121 (+147.62%) |

The absolute H2.50/H2.875 endpoint difference is **0.033906 PPL**, below the
frozen 0.05 flat-curve threshold. Spending another 0.375 bpw on the relative
scales therefore produced no decision-relevant quality improvement: H2.875 is
only 0.121% lower in PPL and both W2 endpoints remain far from W3. This is
accepted negative endpoint evidence, not a completed four-point curve.

Coverage passed for 36 layers, 252 Linears, 6,945,767,424 quantized weights,
and finite decoded weights in both W2 arms. `lm_head` remained BF16. The
same-run W3 replay differs from the historical same-protocol value 11.266115
by only 0.000693 PPL; comparisons above use the same-run value.

BF16 was not replayed by the bounded endpoint job. The 9.724945 row is the
accepted frozen A100 value from the identical token/block protocol and is
shown only as a reference, not as a same-run arm. Against that reference, W3
is 15.85% higher in PPL, while H2.50/H2.875 are 187.23%/186.88% higher.

H2.625 was stopped after 10 partial layers and was never scored. H2.75 was not
started. Neither is a result row. No rotation, distillation, mixed precision,
kernel work, or `lm_head` change was launched.

The result supports the offline diagnosis: relative-scale code width is not
the current bottleneck. The next algorithmic attempt, if authorized, should
first replace the log-domain `q^2` residual fit with an objective aligned to
linear-domain reconstruction MSE, then repeat a cheap projection/endpoint
gate. Do not automatically resume H2.625/H2.75 or launch rotation or
scale-only distillation.

The authoritative artifacts remain on network volume `34au39ljvf`:

```text
/workspace/results/qwen3-8b-hierarchical-w2-v1/H2.50/result.json
/workspace/results/qwen3-8b-hierarchical-w2-v1/H2.875/result.json
/workspace/results/qwen3-8b-hierarchical-w2-v1/ppl-endpoints-result.json
/workspace/jobs/hierarchical-w2-endpoints-20260917T104245Z/
```

The job-produced hashes, structured-result coverage, finite metrics, exit code,
and absence of remaining experiment/GPU processes were checked before shutdown.
The small raw JSON files were not copied to the Mac before the user closed the
server, so this repository records the reviewed decision fields but is not a
second raw-artifact backup. Optional quantization-time and reconstruction-MSE
fields remain in the network-volume JSON and were not used for the decision.

## Endpoint pivot rationale

Offline numerical review found the four relative-scale bit allocations nearly
flat and sometimes non-monotonic. The live four-arm wrapper was therefore
cancelled after H2.50 completed and H2.625 had committed 10 partial layers.
The authorized diagnostic then ran only H2.875 and scored W3, H2.50, and
H2.875. Its absolute endpoint PPL difference was below 0.05, confirming the
flat-curve hypothesis; H2.625/H2.75 remain intentionally incomplete/unscored.

`scripts/run_qwen3_8b_hierarchical_w2_endpoint_job.sh` is the bounded route
used for the final endpoint. The dead-column fail-closed fix was deliberately
not mixed into the two endpoint artifacts; the completed H2.50 artifact had
zero dead columns in all 252 Linears. That defensive fix remains separate
future code work and does not invalidate the reviewed endpoint result.

The original design was a four-point GPTQ quality experiment only. It quantizes the 252
transformer-block Linears, leaves `lm_head` unchanged, evaluates dense BF16
materializations on the frozen WikiText2 protocol, and stops. It does not
build a kernel or launch rotation, QuaRot, SpinQuant, mixed precision, or
distillation.

## Frozen comparison

| Arm | R32 | R16 | Nominal bpw |
|---|---:|---:|---:|
| H2.50 | U4 | U4 | 2.500 |
| H2.625 | U8 | U4 | 2.625 |
| H2.75 | U4 | U8 | 2.750 |
| H2.875 | U8 | U8 | 2.875 |

The direct reference is GPTQ W3 g128: 3.125 semantic bpw, 3.154551630
measured tensor bpw, and historical same-protocol PPL 11.266114820. The new
run scores that same decoded W3 artifact again on the selected device before
the four W2 arms.

This is a quality-only study, so a pinned RTX PRO 4500 Blackwell path is also
available when A100 capacity is unavailable. It is explicitly recorded as
`same-device-quality`, not an A100 reproduction. All four W2 arms and the W3
reference PPL must use that same device and runtime; do not mix A100 and RTX
rows in one curve.

The W2 payload reports both rates. `nominal_bpw` is the formula requested by
the experiment. `actual_bpw_including_permutation_and_steps` additionally
counts the `desc_act` permutation and the two FP16 per-Linear residual steps.
Safetensors headers are not counted as model representation bits.

## Why this is direct GPTQ

For every exact-input group in the frozen order `qkv -> o -> gate/up -> down`,
the runner captures the Hessian after earlier modules in that layer have been
replaced. At each 128-column GPTQ group boundary it fits the hierarchical
projection, fixes its effective scales, then uses `W - W_hat` in the original
column-sequential Hessian compensation loop. There is no ordinary W2 artifact
and no post-hoc scale-fitting stage.

## Reuse the persistent environment

Do not reinstall the dependency stack on each server. The persistent runtime
created by `scripts/prepare_persistent_runtime.py` records the base-image,
Python, CUDA, GPU architecture, lockfile, and project-metadata fingerprint.
On a matching new Pod, source its generated `runtime.sh`; this reuses the venv,
pip cache, Hugging Face cache, and compile caches. Only refresh the cheap
editable project link if the checkout moved:

```bash
source /workspace/runtime/current/runtime.sh
python -m pip install --no-deps -e /workspace/repos/marlin-style-fluxbin
python -m pip check
```

If the base image/runtime fingerprint changes, create a new persistent venv;
do not mutate the old accepted environment. The model snapshot, exact C4
calibration artifact, WikiText2 token artifact, W3 reference, layer payloads,
and results all remain under `/workspace` and survive compute termination.

## Launch

From a clean checkout of the intended commit:

```bash
cd /workspace/repos/marlin-style-fluxbin
source /workspace/runtime/current/runtime.sh

job=/workspace/jobs/hierarchical-w2-$(date -u +%Y%m%dT%H%M%SZ)
output=/workspace/results/qwen3-8b-hierarchical-w2-v1

tmux new-session -d -s hierarchical-w2 \
  "FLUXBIN_PYTHON=$(command -v python) bash scripts/run_qwen3_8b_hierarchical_w2_job.sh /workspace '$job' '$output'"
```

The command above defaults to the formal A100 gate. On the pinned RTX PRO 4500
Blackwell route, launch with the explicit cross-device policy:

```bash
tmux new-session -d -s hierarchical-w2 \
  "FLUXBIN_EXECUTION_POLICY=same-device-quality FLUXBIN_PYTHON=$(command -v python) bash scripts/run_qwen3_8b_hierarchical_w2_job.sh /workspace '$job' '$output'"
```

The RTX result status is `completed_cross_device_pending_review`. The runner
still requires CC 12.0, the frozen package versions, exact artifacts, and the
same-device W3 replay; selecting this policy does not weaken those gates.

The output root may already contain incomplete per-layer artifacts. A new job
directory plus the same output root resumes those layers. The wrapper rejects
a different config hash and skips an arm only when its complete `result.json`
exists. It runs arms one at a time to avoid multiplying model/Hessian memory.

Bounded status checks:

```bash
cat "$job/status"
tail -n 40 "$job/quant-H2.50.log"
tail -n 40 "$job/ppl.log"
```

After successful completion, review:

```text
$output/H2.50/result.json
$output/H2.625/result.json
$output/H2.75/result.json
$output/H2.875/result.json
$output/ppl-result.json
```

`ppl-result.json` contains nominal/effective bpw, PPL, absolute and relative
PPL gaps versus same-run W3, per-arm quantization time, mean weight MSE, and a
report-only monotonicity flag. The current W3 baseline has no additional
accuracy metric under this protocol, so this run does not invent one.

## Decision boundary

The endpoint review is now complete: the relative-scale budget is flat and
neither endpoint approaches W3. Do not complete the two intermediate arms,
add a fifth arm, or automatically launch rotation, scale-only distillation,
kernel work, or a different `lm_head` policy. A new experiment requires a new
projection objective and a separately reviewed protocol.
