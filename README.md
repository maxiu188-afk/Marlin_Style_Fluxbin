# Marlin-Style FluxBin

面向 Qwen3 的低比特权重表示与 packed CUDA decode 研究。当前 QBB M=1
性能链已冻结；同码率质量实验选择 Case A，后续优先评估 uniform 3-bit
backend / solver。vLLM 保留接口但尚未接入。

## 当前结果（2026-09-16）

Qwen3-8B GPTQ W3 inline LUT 的 A100 PCIe 4×3 Linear trial 已完成。按同轮
CUDA Graph total，各 shape 最佳点相对原始 BF16 为：q/o **1.731x**、k/v
**0.982x**、gate/up **2.678x**、down **2.616x**。12/12 cell 正确且稳定，
但 k/v 尚未超过 BF16；当前实例又因 `ERR_NVGPUCTRPERM` 无法采集 Nsight Compute
硬件计数器，所以 prepare 分支仍未授权。详见
[W3 inline 结果](docs/W3_LUT_INLINE_RESULTS.md)。

在 A100 SXM4 80GB 上，`v5_p1024/gps1` + prepared v2.1 的完整模型
32-step CUDA Graph 相对原始 BF16 达到 **1.381x / 1.383x 加速**。

| 固定 prompt | 原始 BF16，32 token | packed，32 token | 速度比 |
|---|---:|---:|---:|
| 0 | 467.513 ms | 338.539 ms | **1.380972x** |
| 1 | 467.919 ms | 338.269 ms | **1.383276x** |

Graph 两组计时均稳定；同轮 eager 超过 5% 波动门槛，不发布加速比，整体记录为
`completed_unstable`。113 项 GPU 测试全部通过，同静态 KV 下包装与 Graph 输出
精确一致。packed/decoded 和动态/静态注意力差异仍 report-only，不宣称数值等价。
这是固定 continuation、真实前缀 KV 的缓存就绪 decode 测量，不是 prefill、自由生成或服务吞吐。

当前 step400 WT2 test PPL 为 **13.169788495**（原始 BF16 **9.724944981**），
较未蒸馏父版本改善 11.9173%，原质量门槛仍未通过。PPL 来自 dense BF16 解码路径，
不能代替 packed 后端的质量验证。完整数值、历史负面结果与证据边界见[结果总览](docs/RESULTS_OVERVIEW.md)。

同协议四臂质量/码率实验进一步得到：GPTQ W3 g128 为 **3.154552 bit/weight、
PPL 11.266115**，当前 QBB 为 **3.138184 bit/weight、PPL 13.167910**。GPTQ
仅多 0.5216% 存储，PPL 低 1.9018（14.4426%），因此当前 QBB point 基本被
uniform W3 支配。QBB FP16-scales 为 2.631687 bit/weight、PPL 13.168951，
保留为低码率 trade-off。该实验在 RTX PRO 4500 上进行，是同卡质量对照而非
A100 复现或性能测试；详见[W3/QBB 状态页](docs/QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

## 阅读与运行入口

- [当前交接](docs/CURRENT_HANDOFF.md)：有效状态、代码入口、服务器与下一步。
- [加速详细结果](docs/QWEN3_8B_M1_LINEAR_RESULTS.md)：版本对照、Linear/block/全模型及哈希。
- [实验运行手册](docs/M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)：当前 prepared v2.1 和历史协议。
- [加速合同](docs/ACCELERATION_HANDOFF.md)：固定输入、数值参照与测量边界。
- [文档索引](docs/README.md)：质量、算法、环境和历史归档。

当前入口为 `scripts/run_qwen3_8b_prepared_m1_trial.py`，配置文件
`configs/acceleration/qwen3_8b_full_m1_v2.json`（协议 ID v2.1）。旧 v1 路径和结果保留。
上一轮 PCIe 的 v1 全模型只有 0.87696x / 0.86674x；GPU 和协议同时变化，
不能将与本轮的差距全部归因为某个 kernel 或 Python 开销。

最后一次服务器验收已完成，证据备份且 GPU 进程退出，可以停止计算实例并保留
网络卷；当前电源状态需下次连接时核验。容器本地环境按 lock 恢复，持久保存模型、
包缓存和兼容的编译缓存，最近恢复约 40 秒。基础镜像 digest 未确认，自建镜像暂缓。
镜像配置及早期构建失败记录见 [RunPod 说明](infra/runpod/README.md)。

以下为历史 32B 算法证据，当前不新增该模型的实验。

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
