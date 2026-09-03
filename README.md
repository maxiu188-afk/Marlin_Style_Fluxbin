# Marlin-Style FluxBin

This repository develops a two-base rank-one binary weight representation and,
after algorithm-quality acceptance, a separate CUDA deployment backend.

## Algorithm arms

The active v2 accuracy path adds the three calibration-dependent mechanisms
that were absent from the first experiment:

- 256 C4 calibration sequences produce `H = 2 X^T X` (stored internally in
  its scalar-normalized equivalent form) and a damped Cholesky inverse;
- hybrid-s8 selects 8 columns per 128-column group using
  `sum_i(W_ij^2) / Hinv_jj^2` before decomposing that group;
- after each group, both arms update all unprocessed columns from the current
  quantization error and the relevant inverse-Hessian blocks.

Pure two-base remains a complete independent arm. It has no residual
refinement, but it does use its own Hessian error-propagation trajectory. The
hybrid arm has a different per-group error after refinement, so its later
global blocks cannot share the pure arm's payload.

The paper specifies C4 and 256 calibration samples but not sequence length or
inverse-Hessian damping. The project freezes those otherwise unspecified
choices at 2048 tokens and 1% mean-diagonal damping and records them explicitly
in the v2 artifacts. Initialization and both ALS iteration limits remain
unchanged at the current greedy initializer and 50 steps.

## Historical v1 weight-only path

The retained global arm approximates each grouped Linear weight block
`W[o,g,j]` as

```text
W_hat[o,g,j] = sum_b row[b,o,g] * column[b,g,j] * base[b,o,g,j],
```

with exactly two `{-1,+1}` bases and group size 128. Each base has its own row
and column factors.

The historical hybrid arm keeps that complete two-base payload and adds sparse residual
refinement. Within every 128-column group, it ranks columns by the global arm's
residual squared error over output rows, selects the stable top 8, and fits a
second independent two-base rank-one decomposition to those residual columns.
The selected indices are stored group-locally in ascending order. This is a
weight-only rule: there is no Shared-C, Hessian propagation, calibration data,
distillation, or deployment kernel at this stage.

Those v1 artifacts remain immutable evidence of the earlier weight-only test;
they are not mixed with v2 calibrated artifacts.

## Evidence ladder

1. Synthetic Hessian, saliency, OBQ propagation, rank-one, and payload tests.
2. Materialize and hash the pinned 256x2048 C4 calibration token artifact.
3. One real Qwen3-32B Linear with independent pure two-base OBQ and
   Hessian-salient hybrid-s8 OBQ outputs.
4. Layer-sequential full-model quantization with both arms, only after manual
   review of stage 3.
5. Dense fake-quantized PPL, only after full-model acceptance.
6. Packed CUDA deployment, separately gated after algorithm-quality acceptance.

No stage launches the next stage automatically.

## Calibrated single-Linear v2 result

Jobs `6271395` and `6271396` completed successfully on Isambard GH200. The
first materialized the pinned 256x2048 C4 token artifact; the second captured
524,288 activation rows for `model.layers.0.self_attn.o_proj.weight` and ran
the two independent calibrated arms.

- Pure two-base OBQ: weight SSE `3088.66483`, calibration output loss
  `2.73732155`.
- Hessian-salient hybrid-s8 OBQ: weight SSE `2819.43539`, calibration output
  loss `1.89756616`.
- Hybrid reduces weight SSE by `8.71669%` and the Hessian-weighted calibration
  loss by `30.67800%` relative to the matched pure arm.
- The two global payloads differ as required, all metrics are finite, packed
  tensor hashes round-trip, and the refinement delta outside selected columns
  is exactly zero.

The calibrated single-Linear execution and artifacts are accepted. Its higher
ordinary weight SSE than historical v1 is not itself a regression verdict:
OBQ changes later working groups to reduce activation-weighted output loss.
Full-model PPL remains the quality gate, and no downstream stage was launched
automatically.

The single-Linear convergence diagnostic also supports a strictly matched
50-versus-200 iteration comparison. It binds the accepted 50-step result by
SHA-256 and changes only `max_iters`; the output records whether the additional
optimization improves SSE beyond the declared `1e-6` relative comparison
tolerance.

The full-model reconstruction is resumable at one payload per Linear. Its
two-base signs use the lossless `fluxbin-two-base-interleaved-2bit-v1` artifact
format, while FP32 scales and group-local `int16` refinement indices remain
explicit. The hybrid payload shares the global arm with the pure result rather
than storing it twice. This compact artifact is for algorithm evaluation and is
not evidence of a Marlin-compatible CUDA layout or runtime speed.

The model-quality gate reuses the accepted QBB-New WikiText-2 token artifact:
146 non-overlapping 2048-token blocks and 298,862 scored next-token
transitions. It evaluates BF16, pure global two-base, and hybrid-s8 sequentially
with BF16 model weights materialized in place. This is dense fake-quant PPL,
not packed-kernel execution.

The accepted execution from job `6259037` reproduced BF16 PPL `7.61084`. Pure
global two-base rank-one produced PPL `147.53152`; hybrid-s8 improved it to
`24.21529`, but remained `3.18x` the BF16 PPL. The run and provenance are valid,
while the algorithm-quality result is negative; these weights do not authorize
the packed-backend stage.

## Local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

The current macOS host has no project PyTorch environment and no NVIDIA GPU.
Formal tensor experiments run on Isambard GH200 through Slurm.
