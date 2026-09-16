# GPTQ W3 inline LUT：完整模型结果与误差归因

更新：2026-09-16。本文记录 Qwen3-8B、batch 1、M=1 decode 的 W3 inline LUT
完整模型 trial。正式状态为 `completed_with_backend_numerical_differences`：性能结果稳定，
但 packed-vs-decoded W3 数值门未通过，因此**不是完整模型 correctness accepted 结果**。

## 结论

在 NVIDIA A100 80GB PCIe 上，完整 36 层、252 个 W3 Linear 全部走 packed route。
同轮 32-token CUDA Graph total 相对原始 BF16 达到 **1.4880x / 1.4892x**，相对
decoded W3 BF16 达到 **1.4923x / 1.4886x**。两组主指标都稳定，wall-time
相对极差低于 0.1%。

但 packed-vs-decoded W3 的 logits NRMSE 为 **0.01683 / 0.01457**，超过冻结的
0.005；最大 log-probability 误差为 **0.59375 / 0.59824**，超过 0.05。两组
greedy tokens 和 fed tokens 相同，只说明本次固定 continuation 没有 token 分叉，
不能替代数值门。

因此当前证据支持“W3 inline kernel 在完整模型 decode 上有约 1.49x 性能潜力”，
不支持“它已严格复现 retained decoded-BF16 GPTQ 模型”。

## 冻结协议与性能

- 模型：`Qwen/Qwen3-8B` revision
  `b968826d9c46dd6066d109eabc6255188de91218`。
- W3：symmetric GPTQ W3 g128、36 层、252 个 Linear；layout manifest SHA256
  `f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8`。
- candidate：q/o R256、k/v R512、gate/up R1024、down R1024，来自已完成的
  [4x3 Linear trial](W3_LUT_INLINE_RESULTS.md)。
- 三臂：原始 BF16、decoded W3 BF16、packed W3 inline；全部模型、cache 和 Graph
  同时驻留，allocated 38,456,755,200 bytes，setup peak 49,268,260,864 bytes。
- workload：两个真实 prompt、真实前缀 StaticCache、32 个固定 continuation token。
  Graph 包含 embedding、全部 36 个 block、LM head、argmax 和 W3 kernel；load、转换、
  prefill、KV reset、Graph capture、审计和 CPU copy 不计时。
- 正式指标：同一次运行的 full-sequence CUDA Graph wall total；10 次交错测量。

| prompt | original BF16 | decoded W3 BF16 | packed W3 | vs original | vs decoded W3 |
|---|---:|---:|---:|---:|---:|
| 0 | 477.097 ms | 478.485 ms | 320.625 ms | **1.48802x** | **1.49235x** |
| 1 | 479.077 ms | 478.903 ms | 321.710 ms | **1.48916x** | **1.48862x** |

两组 Graph 的三臂均稳定。prepared eager 中存在超过冻结门限的 cell，因此不发布
eager 加速比，也不把 `primary_timings_stable=true` 写成所有 mode 都稳定。

## 数值门与逐层定位

| prompt | logits NRMSE | limit | max abs logits | max abs log-prob | greedy/fed tokens |
|---|---:|---:|---:|---:|---|
| 0 | 0.0168270 | 0.005 | 0.59375 | 0.59375 | equal / equal |
| 1 | 0.0145725 | 0.005 | 0.43750 | 0.59824 | equal / equal |

prompt 0 的 decode-step-0 定位结果如下：

- packed 和 decoded 的 prefill logits、36 层 prefix key/value cache 均逐位一致。
- 第 0 层 block 输出 NRMSE 已为 0.001506；第 8 层首次超过 0.005，第 21 层约
  0.014459，第 35 层为 0.008594；该步最终 logits 为 0.008560。
- 把每个 packed Linear 捕获到的**同一输入**送入对应 decoded dense Linear 后，
  252 个 Linear 的本地 NRMSE 都只在约 0.00059--0.00264。七类模块中位数为
  0.00152--0.00182，没有某个 shape 或投影单独异常。
- prepared wrapper、CUDA Graph、路由和固定 continuation 的精确检查均通过；
  所以误差不是 StaticCache 起点、Graph capture 或 dense fallback 引起。

## 当前算术定位：语义差异已确认，具体主导项尚未闭环

decoded 路径先逐元素物化 `BF16(scale * signed_code)` 权重，再执行 BF16 dense
GEMM。inline kernel 则先以 FP32 计算组内 integer dot，再乘 BF16 scale，并在最终
输出处转回 BF16。这两条路径在实数算术中等价，在有限精度下不再严格等价。

对 layer 0 的四种真实 shape 做了额外算术分解：

| module / shape | packed vs structural BF16 | 仅 FP32 归约顺序 NRMSE | 权重提前 BF16 物化 NRMSE | dense backend vs FP32 NRMSE | packed vs dense BF16 |
|---|---:|---:|---:|---:|---:|
| q `[4096,4096]` | **0** | 1.28e-7 | 7.08e-4 | 3.18e-5 | 0.002049 |
| k `[1024,4096]` | **0** | 1.41e-7 | 6.28e-4 | 0 | 0.002178 |
| gate `[12288,4096]` | **0** | 1.31e-7 | 6.50e-4 | 7.46e-5 | 0.001533 |
| down `[4096,12288]` | **0** | 3.31e-7 | 3.41e-4 | 4.29e-5 | 0.000650 |

packed 与 structural reference 在四种 shape 上仍逐位一致；第一轮 12 个 Linear
cell 也全部 structural exact。这排除了已测试输入上的 `g_idx/desc_act`、planar
packing 和 LUT bit-plane 解码错误。表中各 NRMSE 使用不同中间精度和最终 cast，
不是可以直接相加的因果分解；尤其 1e-7 量级的 FP32 差异仍可能在 BF16 边界触发
离散 ULP 变化。因此当前只能确认问题位于 dense 权重物化、归约顺序和最终 BF16
舍入之间，不能把 per-weight BF16 舍入单独写成已证实根因。

后续已实现独立的 `±3` BF16 舍入修正：对 BF16 scale 和 signed W3 code，只有
`q=+3/-3` 需要额外舍入 correction，可直接由已有 bit-plane 构造两个 mask 而无需
增加 payload。该候选仍等待 NVIDIA GPU 上的真实 Linear 因果门和性能测量，详见
[BF16/split-G 诊断手册](W3_LUT_BF16_SPLIT_RUNBOOK.md)。

## 决策边界

当前不能放宽阈值后把本轮改写为 accepted，也不能用两个 prompt 的 greedy token
一致代替质量验证。下一步应先明确语义：

1. 如果目标是严格复现 retained decoded-BF16 GPTQModel，需要让 kernel 模拟
   per-weight BF16 物化/舍入，再重做 Linear 和完整模型正确性、性能门。
2. 如果保留当前 structural W3 算术，应把它视为另一个部署模型，单独完成完整
   PPL/质量验证；它不能直接通过当前 decoded-W3 oracle。

Nsight Compute 在该宿主仍因 `ERR_NVGPUCTRPERM` 不可用。prepare 不是当前优先项；
在数值语义决策前，不自动继续 profiler、prepare、多 batch 或 serving。

## 来源与保存

- 执行源码 revision：`fc76b75802d57e238eaf258ab85f3803588e92c8`。
- protocol SHA256：`a82aec5904b29b391562dd45872b5955160191ce52be23d66899085fdceab127`。
- runner SHA256：`cadb928f6b2f8732529da3c950838d65d57864f06ba40857fed0f30d1015bd2b`。
- environment JSON SHA256：`0e75d87e3fe7c2de4525cd0b3349656dae2088ad6778a017c3c1be11f710d558`。
- 正式结果 JSON SHA256：`c68811aa065b34d497f3113e0bc873684624ee0c8fc8df199998a7e53856f423`。
- 逐层归因 JSON SHA256：`8620846807f7214912c752aeb79d6011f61892d71dd1c22a81e96ca96d87cbe6`。
- 算术分解 JSON SHA256：`f1767d0e7e673a31923573eafa9c38673c3698fec61abc1e653fe2a386a64a85`。

远端结果位于 `/workspace/results/qwen3-8b-w3-full-m1-v1/`；私有本地备份位于
`server_results/runpod_w3_full_m1_a100_pcie_2026-09-16/`，不提交 GitHub。
最后检查时无 GPU compute 进程或 tmux，会话和结果均已收尾；计算实例可以关闭，
但应保留 `/workspace` 网络卷。
