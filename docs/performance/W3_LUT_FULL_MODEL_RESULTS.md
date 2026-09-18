# GPTQ W3 inline LUT：完整模型结果与误差归因

更新：2026-09-17。本文记录 Qwen3-8B、batch 1、M=1 decode 的 W3 inline LUT
structural 与 decoded-BF16 corrected 完整模型 trial。两次正式状态都是
`completed_with_backend_numerical_differences`，因此都不是完整模型 correctness
accepted 结果；性能结论分别按各自计时门解释。

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

## 2026-09-17 corrected follow-up：性能优先结论

同型 NVIDIA A100 80GB PCIe 上又完成四臂同轮对照：original BF16、decoded W3
BF16、`fast_corrected` 和 `observed_exact`。四臂及其 cache/Graph 同时驻留，allocated
43,840,886,784 bytes，setup peak 54,803,371,008 bytes。下表是 10 次交错测量的
CUDA device median；每个值都对应 32-token full-sequence Graph。

| prompt | original BF16 | decoded W3 BF16 | fast corrected | fast vs original | observed exact | observed vs original |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 476.338 ms | 476.319 ms | 354.399 ms | **1.34407x** | 374.968 ms | **1.27034x** |
| 1 | 476.726 ms | 476.716 ms | 355.055 ms | **1.34268x** | 375.391 ms | **1.26995x** |

`fast_corrected` 相对 decoded W3 为 1.34402x / 1.34265x，约 90.29 / 90.13
token/s；`observed_exact` 为 1.27029x / 1.26992x，约 85.34 / 85.25 token/s。
两条 corrected 路线都没有达到 1.5x，也都慢于历史 structural 运行的
320.625 / 321.710 ms、1.4880x / 1.4892x。跨运行比较不是单因素消融；按中位数估算，
fast corrected 比 structural 慢约 10%，observed exact 慢约 17%。因此只看速度时，
structural 路线仍是当前 W3 主线，corrected 路线不应替代它。

正式 JSON 没有发布 speedup 字段，因为冻结的 max-min stability gate 看到三个孤立
outlier：prompt 0 decoded W3 有一次 511.310 ms，prompt 1 original BF16 有一次
558.427 ms，prompt 1 observed exact 有一次 452.869 ms；wall-time 还存在 host-side
outlier。去掉各组 min/max 后，八个 CUDA device timing cell 的相对极差均不超过
0.26%；`fast_corrected` 自身两组 CUDA device 计时也通过 5% 门，但 wall-time 未通过。
因此表中数值是**可复核的 raw CUDA median 性能观测**，不是 formal-stable comparison
acceptance。

correctness 没有随修正通过：

| route | prompt 0 logits NRMSE / max log-prob | prompt 1 logits NRMSE / max log-prob | accepted |
|---|---:|---:|---|
| fast corrected | 0.020764 / 1.09243 | 0.022001 / 1.12491 | false |
| observed exact | 0.019733 / 0.71975 | 0.017023 / 0.95927 | false |

冻结门限为 logits NRMSE 0.005、max log-prob 0.05。两条路线的 prepared wrapper、
Graph 检查均精确；每个 prompt 都捕获 252 条 packed route，`dense_fallback=0`，
greedy/fed tokens 也一致。因此 failure 不是 dispatch、Graph 或 fallback 问题。
`±3` correction 能复现逐权重 BF16 物化后的显式 FP32 grouped matvec，但 dense
oracle 是 cuBLAS BF16 GEMM，其 reduction tree 与 kernel 的“先 group dot、后乘 scale、
再 split reduction”不同。实数公式相同，浮点执行顺序不同；RTX layer-0/fixed-input 的
最终 BF16 coincidence 不能外推为全输入、全层 bit-exact，36 层传播后差异继续放大。

W3 的有效存储为 3.154552 bit/weight，相对 BF16 是约 5.07x，但相对理想 W4 只有
约 1.27x。A100 没有原生 INT3 MMA；当前路径还要做 3 个 bitplane 解码、activation
LUT、shared-memory build/barrier、permutation gather、FP32 partial workspace 和 finish
reduction。gps1 会产生 32 或 96 个 partial split；252 个 Linear × 32 token 又有
8064 次 main/finish 调用。即使整段被 CUDA Graph 捕获，device-side launch/dependency、
非 Linear 模型工作和 reduction 流量仍存在。相比成熟的 Tensor Core-friendly W4A16，
W3 的额外开销只要超过约 27%，就会吃掉其相对 W4 的理论存储优势。

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

后续已实现并完成独立的 `±3` BF16 舍入修正诊断：对 BF16 scale 和 signed W3 code，只有
`q=+3/-3` 需要额外舍入 correction，可直接由已有 bit-plane 构造两个 mask 而无需
增加 payload。RTX PRO 4500 的 46/46 cell 均通过 correctness、repeat 与 Graph 检查；
gps1 对 q/o、k/v 达到 fixed-input bit-exact，对 gate/up 仅余约 4.01e-8 NRMSE，down
从 0.001821 降至 0.000154。gate/up gps4、down gps2 在该输入上进一步达到 bit-exact。
这些结果支持把修正路线接入完整模型，但并不是跨输入根因闭环。上面的 A100
follow-up 已进一步证明 fixed-input exact 不能外推为完整模型 exact。诊断
result SHA256 为
`4f567a9adf28d80eb2f7d08f08146a08a855d80f3c9589f02d5bd845f51460ca`；完整执行记录见
[A100 corrected full-model 手册](W3_CORRECTED_FULL_MODEL_RUNBOOK.md)。

## 决策边界

不能放宽阈值后把任一轮改写为 accepted，也不能用两个 prompt 的 greedy token 一致
代替质量验证。但 0.005 这道门本身已被标定为不可达：零量化的 `original_bf16` 臂在一次
数学等价的 mask 改写下就落在 0.01431 / 0.01153，QBB 线独立复现同一地板。详见
[数值门标定](W3_NUMERICAL_GATE_CALIBRATION.md)；继续按现行门限迭代 kernel 只会产生
误导性的负面结果。用户当前明确以加速效果为主，所以性能主线保留 structural W3；它仍须
被视为另一个部署算术语义，后续若要发布质量结论，应单独完成完整 PPL/质量验证。
corrected 两条路线冻结为负面 follow-up，不自动重跑。

下一轮性能工作应优先在允许 performance counters 的实例上 profile structural 路线，
核实 main/finish fusion、partial workspace、LUT build/barrier 和非 Linear 固定时间的
占比，再决定 persistent/fused Linear 或 Tensor Core-friendly W3 重构。现有宿主的
Nsight Compute 因 `ERR_NVGPUCTRPERM` 不可用；不自动扩展 batch、serving 或 prepare。

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

corrected follow-up 的附加来源：

- 执行源码 revision：`d0f85b49745d169b6ffc1f4a7e4d97a197e6911d`。
- protocol SHA256：`f7d3f12cc29a645e1d9862c4fc245a8ffc32b22428136e6b8c86427b55256b24`。
- runner SHA256：`511a9ac78e3fc5437ed19833fe4b71f47c2899798503197871e0dd163547054e`。
- environment JSON SHA256：`9f045ce580131aac8f717d53ea7fe1a85c5c8f389c5fce0111ce30afed23091e`。
- 正式结果 JSON SHA256：`08271b47c7db3e5197557a6fef25af659cf90e885621e7d4660a99d3c3a2c0cd`。

远端 corrected 结果位于
`/workspace/results/qwen3-8b-w3-corrected-full-m1-v1-a100-pcie-20260917-retry1/`；
私有本地备份位于
`server_results/runpod_w3_corrected_full_m1_a100_pcie_2026-09-17/`，不提交 GitHub。
retry1 退出 0，最后审计 GPU、tmux 和实验进程均为空；服务器已经 shutdown-ready，
实际电源状态仍由用户确认。
