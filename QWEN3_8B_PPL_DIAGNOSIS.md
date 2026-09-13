# Qwen3-8B PPL diagnosis — offline evidence, 2026-09-13

A concrete sequential OBQ compensation defect is confirmed by a CPU algebraic
counterexample. It is shared by the 8B and historical 32B implementations.
Its contribution to full-model PPL has not yet been measured. No algorithm code,
weights, quantization runs or PPL runs were changed by this diagnosis.

## Compare the correct baselines

| Source / model | Baseline PPL | Pure PPL | Hybrid-s8 PPL |
| --- | ---: | ---: | ---: |
| Our Qwen3-8B | 9.724945 | 1149.470625 | 16.142104 |
| Our Qwen3-32B | 7.610839 | 17.744707 | 10.433873 |
| FluxBin PDF, Table 5, Qwen3-8B | 9.72 (FP16) | not reported in Table 5 | 13.46 |

The PDF also reports 8B s16 at 12.44. Its pure PPL 13.71 (Tables 2/6) is
LLaMA-2-7B, not Qwen3-8B. Table 5 was visually checked on PDF page 17.
Our s8 is about 19.93% above the paper's s8; comparing our pure 1149 to its
hybrid 13.46 mixes configurations. BF16 versus the paper's FP16, exact
calibration sample provenance and unspecified implementation details remain
comparison limitations despite the closely matching baseline PPL.

Historical 32B hybrid is already 37.09% above its own baseline. Paper 8B s8 is
38.48% above its baseline. Thus even reproducing the paper would not satisfy
our independently chosen 5% gate. That gate cannot be used as a definition of
paper reproduction. It has not been changed by this diagnosis.

## Confirmed defect: stale inverse Hessian across groups

`src/fluxbin_style/hessian_obq.py` computes the full inverse Hessian once.
Both quantization loops then slice that same inverse at every group (pure
lines 315–320; hybrid lines 433–438). They update the remaining weights but
never condition the inverse on previously fixed groups.

For a group B and remaining free coordinates R, let C be the inverse Hessian
of the *current free subproblem*. The valid block compensation is:

```text
update_R = error_B @ inverse(C_BB) @ C_BR
C_next   = C_RR - C_RB @ inverse(C_BB) @ C_BR
```

The first group may use the original full inverse. After fixing that group,
the free-subproblem inverse is C_next, not the corresponding raw slice C_RR.
The existing block solve is correct when supplied the current inverse; the
caller's reuse of the original inverse is the defect. An equivalent stable
Cholesky formulation can avoid explicit repeated Schur updates, but it must
preserve this conditional-subproblem invariant.

Local counterexample uses the actual `obq_error_update` on CPU with:

```text
C = [[2.0, 0.8, 0.6], [0.8, 1.5, 0.9], [0.6, 0.9, 1.2]]
```

After coordinate 0 is fixed and the next coordinate has unit residual:

| Method | Compensation | Free-coordinate stationarity residual |
| --- | ---: | ---: |
| Existing raw inverse slice | 0.6000000 | 0.0625001 |
| Inverse of remaining Hessian | 0.5593221 | 0.000000060 |

The existing update is not the minimizer of the remaining quadratic problem.
The corrected update reduces this local quadratic loss from 0.8500000 to
0.8474576. This small example demonstrates the invariant failure; it does not
predict an 8B PPL improvement or explain the entire observed degradation.
Evidence is saved in `server_results/diagnostics/obq-conditioning-counterexample.json`;
the reproduction script is `tmp/verify_obq_conditioning.py`.

The existing scalar compensation test covers only the first step. Other tests
cover identity Hessians, finite reconstruction and payload support, not a
correlated multi-step conditional optimum. Passing them, or matching the
historical implementation, did not validate this invariant. Earlier artifact
and PPL execution acceptances remain records of the produced results, not proof
that the sequential algorithm is mathematically correct.

The official GPTQ implementation uses an additional upper Cholesky factor of
the inverse for its sequential updates:
https://github.com/IST-DASLab/gptq/blob/main/gptq.py . The FluxBin PDF's
Algorithms 1/2 (page 13) abbreviate compensation to an unspecified update;
that pseudocode alone does not establish which conditioning strategy the
authors' implementation uses. This finding is an OBQ correctness issue, not a
claim to have verified the authors' exact code.

## Where the 8B branches diverge

The complete 8B metadata is locally archived and was inspected without server
access. Zero-based layer indices:

| Diagnostic | Pure | Hybrid-s8 |
| --- | ---: | ---: |
| Layer 1 gate_proj calibration output SSE | 2122.306 | 206.928 |
| Layer 2 gate_proj calibration output SSE | 2805.455 | 331.977 |
| Layer 6 down_proj calibration output SSE | 12688.900 | 41.865 |
| Layer 6 down Hessian damping | 0.951494 | 0.021442 |

Damping is 1% of mean Hessian diagonal, and H = 2 X^T X / activation_rows.
Therefore the last row indicates roughly 44.38x mean squared input activation
in pure's layer-6 down input relative to hybrid's. This is a location for
investigation, not a matched-input comparison: earlier quantized layers already
produce different branch inputs. It does not show 44x growth versus BF16,
and it does not isolate whether the cause lies in layer 6 or earlier layers.

Global weight relative Frobenius errors are close across model sizes:

| Arm | 8B | 32B |
| --- | ---: | ---: |
| Pure | 0.384708 | 0.377090 |
| Hybrid-s8 | 0.360005 | 0.358877 |

A modest global reconstruction difference accompanies an enormous pure PPL
difference. Average weight SSE hides functional sensitivity and propagation
through nonlinear MLPs/attention. The evidence points toward early MLP/activation
sensitivity rather than a uniformly much worse weight fit. The same defect
can have model-dependent impact; better 32B PPL does not exclude it.

A second, lower-confidence issue is calibration order inside a transformer
layer. `run_qwen3_8b_full_hessian_obq_s8.py:618` captures all Hessians before
quantizing any module in that layer, then applies all replacements. Thus o_proj
is calibrated with unquantized current-layer q/k/v, and down_proj with
unquantized current-layer gate/up. Subsequent inference uses quantized versions.
This is a frozen design choice shared with 32B, not a proven paper mismatch.
A true-sequential qkv → o → gate/up → down capture would test its impact after
the compensation issue is isolated.

## Recommended next sequence (not executed)

1. Add a correlated multi-group KKT/Schur-complement regression test and fix
   only the compensation invariant in a new version. Preserve existing artifacts.
2. With access to the retained server data, test a bounded early-layer target
   (especially gate/up and layer-6 down) with identical W/X and frozen parameters,
   comparing old versus corrected compensation. Recompute real output loss.
3. Only after that controlled comparison, rerun the 8B pure/s8 full pipeline and
   matched PPL. Do not change group size, calibration, damping, fit iterations
   and within-layer ordering simultaneously.
4. If the paper gap persists, separately test within-layer sequential capture
   and check author implementation/calibration details. Keep paper reproduction
   and the stricter deployment-quality target as separate decisions.

No claim is made that increasing sample count, adding distillation, switching
GPU, or merely using a larger model will fix the result. Server work was not
started; the user is retaining server storage for later recovery.

## Hybrid-focused follow-up, 2026-09-13

User direction: focus on hybrid; pure is not an optimization target. Investigate
implementation/quality gaps first, and consider distillation later. No new GPU
work, fitting changes, or distillation has been launched.

### Compensation and selection must be separated

The compensation defect above is a conditional quadratic-optimum violation.
Changing the saliency definition is a distinct algorithm decision: the paper's
Eq. 7 uses the inverse Hessian diagonal, while its abbreviated groupwise
pseudocode does not establish whether author code conditions it between groups.
Do not silently change both compensation and selection in a single “bug fix.”

The actual hybrid function was instrumented on a synthetic CPU case (seed
20260902, W shape [24,128], four groups of 32, two selected columns per group,
eight fitting iterations). It returned the original selections and weights;
only a parallel diagnostic recomputed selections on each same working weight
with the inverse of the remaining Hessian. Selections, group-local indices:

| Group | Existing full-inverse slice | Remaining-subproblem inverse |
| --- | --- | --- |
| 0 | [2,19] | [2,19] |
| 1 | [17,18] | [16,18] |
| 2 | [14,15] | [2,15] |
| 3 | [20,24] | [10,20] |

This proves the choice can alter which columns receive hybrid refinement. It
does not show that the conditioned selector has lower real-model loss, or that
it is the authors' implementation. The example is not an 8B benchmark.
Evidence: `server_results/diagnostics/hybrid-selection-conditioning.json`;
reproduction: `tmp/check_hybrid_selection.py`.

### Fitting budget is another measurable suspect

Across all 10,368 hybrid groups, the recorded stopping reasons are:

| Fit | Relative-tolerance stop | Reached 50-iteration cap |
| --- | ---: | ---: |
| Global two-base fit | 5039 | 5329 (51.40%) |
| Sparse refinement fit | 9591 | 777 (7.49%) |

The global fit uses 436,263 total iterations and refinement 308,778. Several
middle-layer gate/up projections hit the global cap in every group. Hitting
the cap does not prove that extra iterations improve PPL: the full artifacts
do not retain per-iteration loss curves. Before allocating a longer full run,
inspect a bounded same-W/X target with unchanged initialization and record the
loss trajectory beyond 50 iterations. Do not claim the paper used more
iterations; its exact runtime setting has not been established here.

### Author source availability

The paper points to `https://github.com/nicyyyy/FluxBin`. GitHub API lookup
returned HTTP 404 in this session. This means the referenced repository was
not accessible through this lookup, not that the authors never published code.
No alternate or third-party implementation has been treated as canonical.

### Order of controlled follow-ups

1. Hybrid only: validate the compensation fix against multi-group quadratic
   oracles, holding the existing saliency policy fixed as an explicit variant.
2. On identical real W/X, separately compare the saliency policies. Record
   index overlap, output loss and payload identity, not just weight SSE.
3. Separately test within-layer sequential calibration and iteration budget;
   do not bundle them with the compensation change.
4. Evaluate the resulting hybrid with the same matched PPL protocol before
   deciding a distillation scope. Preserve the current 16.142104 artifact as
   the comparison baseline. No pure refit is necessary for this question.

A validated correction should precede distillation: otherwise training could
compensate for a defect without explaining the paper-reproduction gap. Neither
paper-level quality nor the stricter deployment target is guaranteed by these
checks.
