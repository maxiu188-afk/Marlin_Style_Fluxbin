# Marlin-Style FluxBin

This repository develops a two-base rank-one binary weight representation and,
after algorithm-quality acceptance, a separate CUDA deployment backend.

## Frozen phase-1 algorithm

For each grouped Linear weight block `W[o,g,j]`, phase 1 approximates

```text
W_hat[o,g,j] = sum_b row[b,o,g] * column[b,g,j] * base[b,o,g,j],
```

with exactly two `{-1,+1}` bases and group size 128. Each base has its own row
and column factors. There is no shared column, salient-column refinement,
compensation matrix, distillation, Hessian propagation, or deployment kernel in
this phase.

FluxBin supplies only the two-base row-column decomposition idea. Its reported
accuracy, calibration, kernel, and end-to-end claims are not acceptance targets
for this project.

## Evidence ladder

1. Synthetic algebra, determinism, monotonicity, and payload tests.
2. One real Qwen3-32B Linear reconstruction gate.
3. Full-model reconstruction, only after manual review of stage 2.
4. Dense fake-quantized PPL, only after stage 3 acceptance.
5. Packed CUDA deployment, separately gated after algorithm-quality acceptance.

No stage launches the next stage automatically.

## Local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

The current macOS host has no project PyTorch environment and no NVIDIA GPU.
Formal tensor experiments run on Isambard GH200 through Slurm.
