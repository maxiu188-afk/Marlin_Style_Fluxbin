# Marlin-Style FluxBin

Hardware-aware research on ultra-low-bit LLM weight representations and
packed CUDA decode. The project studies the complete path from quantization
quality and effective bits per weight to real Linear kernels, CUDA Graph
execution, operator fusion, and full-model batch-1 decoding.

The current focus is Qwen3-8B. Earlier Qwen3-32B experiments established the
algorithmic reconstruction pipeline and motivated the move from flexible
binary-basis formats toward more regular 2--3 bpw representations.

## What I built

- A two-base binary representation with rank-one scaling, optional salient
  residual columns, and scale-only post-quantization distillation.
- Full-model Qwen3-8B and Qwen3-32B reconstruction pipelines with frozen
  calibration, payload manifests, and WikiText-2 evaluation.
- Packed lookup-table CUDA kernels for real Qwen3 Linear shapes, including
  layout preparation, scale fusion, split reductions, and CUDA Graph capture.
- A regular GPTQ W3 backend using inline activation lookup and packed 3-bit
  weights.
- Full-model batch-1 decode with real prefix KV cache, plus fused RMSNorm and
  RoPE implementations.
- Profiling and result tooling that separates representation quality, kernel
  speed, full-model speed, memory cost, and serving readiness.

## Selected results

| Study | Main result |
|---|---|
| Qwen3-8B binary-basis quality | Scale-only distillation improved WikiText-2 PPL from **14.95 to 13.17** |
| Matched-rate comparison | GPTQ W3 reached **11.27 PPL at 3.155 bpw**, versus **13.17 at 3.138 bpw** for the binary-basis format |
| W3 Linear kernels | Q/o **1.73x**, k/v **0.98x**, down **2.62x**, and gate/up **2.68x** over same-run BF16 on A100 PCIe |
| W3 full-model decode | Packed batch-1 decode reached about **1.49x** over original BF16 on A100 PCIe |
| Non-Linear fusion | Fused RMSNorm/RoPE raised the representative packed full-model point to **1.62x** over BF16 on A100 SXM4 |
| Binary-basis full model | The prepared lookup-table route reached **1.38x** over original BF16 for 32-token decode |
| Historical Qwen3-32B quality | Hybrid residual refinement improved PPL from **17.74 to 10.43** relative to the pure two-base arm |

All speedups above are measured against the corresponding BF16 arm in the
same experiment. Exact protocols, hardware variants, result JSONs, and
validation status are recorded in the linked result documents.

## Main findings

### 1. Regularity is part of the quantization objective

At almost identical storage, the regular 3-bit representation was both more
accurate and easier to accelerate than the flexible binary-basis format. This
shifted the project from adapting kernels to an irregular representation
toward designing the representation around efficient GPU execution.

### 2. Kernel speed is only one part of full-model speed

Real-shape W3 Linear kernels reached up to 2.68x, while full-model decode was
about 1.49x. Profiling showed that Linear layers accounted for approximately
62.1% of the BF16 decode time; the LM head and many small non-Linear kernels
formed the remaining ceiling.

### 3. Runtime structure matters

Memory layout, lookup construction, scale placement, launch count, CUDA Graph
capture, and fusion changed performance without changing the nominal bit
width. Fusing RMSNorm and RoPE improved both the BF16 and packed paths and
raised the packed/BF16 ratio to 1.62x in the representative fused run.

## Current research direction

The next representation starts from a regular 2-bit computational core and
uses a bounded auxiliary budget of roughly 0.5--1 bpw only where it provides
the most accuracy. The target is W3-level model quality at 2.5--3 bpw with:

- fixed, contiguous codes for coalesced GPU loads;
- no irregular sparse indexing in the critical path;
- explicit storage and decode cost for every auxiliary field;
- quality evaluation before backend promotion;
- a direct path to packed operator and full-model execution.

Early hierarchical-W2 and rotation probes narrowed the search: simply adding
more scale precision was ineffective, so the auxiliary bits must encode more
decision-relevant weight structure.

## Repository structure

```text
src/fluxbin_style/  Quantizers, payloads, model adapters, CUDA wrappers,
                    and fused non-Linear modules
configs/             Calibration, quality, acceleration, and evaluation
                     protocols
scripts/             Reconstruction, PPL, GPU benchmark, profiling, and
                     recovery entry points
tests/               Local algebra tests and GPU regression coverage
docs/                Current handoff plus quality, performance, operations,
                     and archive sections
infra/runpod/        Reproducible RunPod environment and cache recovery
results/             Local compact results
server_results/      Private Git-ignored evidence backup
```

Large model weights and private server evidence are deliberately excluded
from Git. The public repository keeps source, frozen configs, documentation,
and compact reproducibility records.

## Local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

Local macOS checks cover source, algebra, and small synthetic cases. CUDA
performance results come from the NVIDIA environment named in each result
document.

## Start here

1. [Current handoff](docs/CURRENT_HANDOFF.md) -- current state, retained
   artifacts, and the active research direction.
2. [Results overview](docs/RESULTS_OVERVIEW.md) -- consolidated quality and
   full-model results.
3. [Documentation index](docs/README.md) -- topic-level navigation across
   quality, performance, operations, and archive material.
4. [Quality documentation](docs/quality/README.md) -- reconstruction,
   distillation, rate-distortion, W2, and rotation studies.
5. [Performance documentation](docs/performance/README.md) -- Linear kernels,
   full-model decode, numerical analysis, and fusion.
6. [Operations documentation](docs/operations/README.md) -- RunPod setup,
   recovery, and server workflow.
7. [W3 Linear results](docs/performance/W3_LUT_INLINE_RESULTS.md),
   [W3 full-model results](docs/performance/W3_LUT_FULL_MODEL_RESULTS.md), and
   [non-Linear fusion](docs/performance/NONLINEAR_FUSION.md) -- the main CUDA
   acceleration evidence.
8. [W3/QBB rate-distortion result](docs/quality/QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)
   -- the quality result that motivated the current regular representation.
