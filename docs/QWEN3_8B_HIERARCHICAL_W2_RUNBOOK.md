# Qwen3-8B hierarchical-scale W2 quality runbook

This is a four-point GPTQ quality experiment only. It quantizes the 252
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

Review only whether PPL improves stably over the four budgets and where it
approaches W3. Do not add a fifth arm or automatically launch rotation,
scale-only distillation, kernel work, or a different `lm_head` policy. Those
are separate decisions after this curve is reviewed.
