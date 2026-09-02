# Marlin-Style FluxBin

This repository develops a two-base rank-one binary weight representation and,
after algorithm-quality acceptance, a separate CUDA deployment backend.

## Algorithm arms

The retained global arm approximates each grouped Linear weight block
`W[o,g,j]` as

```text
W_hat[o,g,j] = sum_b row[b,o,g] * column[b,g,j] * base[b,o,g,j],
```

with exactly two `{-1,+1}` bases and group size 128. Each base has its own row
and column factors.

The hybrid arm keeps that complete two-base payload and adds sparse residual
refinement. Within every 128-column group, it ranks columns by the global arm's
residual squared error over output rows, selects the stable top 8, and fits a
second independent two-base rank-one decomposition to those residual columns.
The selected indices are stored group-locally in ascending order. This is a
weight-only rule: there is no Shared-C, Hessian propagation, calibration data,
distillation, or deployment kernel at this stage.

FluxBin supplies only the two-base row-column decomposition idea. Its reported
accuracy, calibration, kernel, and end-to-end claims are not acceptance targets
for this project.

## Evidence ladder

1. Synthetic algebra, determinism, monotonicity, and payload tests.
2. One real Qwen3-32B Linear pure two-base reconstruction gate.
3. One real Qwen3-32B Linear hybrid-s8 reconstruction gate.
4. Full-model reconstruction with both pure two-base and hybrid-s8 outputs,
   only after manual review of stage 3.
5. Dense fake-quantized PPL, only after full-model acceptance.
6. Packed CUDA deployment, separately gated after algorithm-quality acceptance.

No stage launches the next stage automatically.

The full-model reconstruction is resumable at one payload per Linear. Its
two-base signs use the lossless `fluxbin-two-base-interleaved-2bit-v1` artifact
format, while FP32 scales and group-local `int16` refinement indices remain
explicit. The hybrid payload shares the global arm with the pure result rather
than storing it twice. This compact artifact is for algorithm evaluation and is
not evidence of a Marlin-compatible CUDA layout or runtime speed.

## Local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

The current macOS host has no project PyTorch environment and no NVIDIA GPU.
Formal tensor experiments run on Isambard GH200 through Slurm.
