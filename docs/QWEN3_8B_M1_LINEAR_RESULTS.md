# Qwen3-8B 首轮 M=1 Linear 结果

2026-09-14：**真实权重的数值门槛通过，但首版 kernel 未实现加速**。
CUDA Graph 轮七项计时均稳定，kernel 耗时为同轮 dense BF16 的 2.03–2.26 倍。
这是保留的优化基线，不是 block/full-model/vLLM 或产品部署结果。

## 冻结范围

- A100 80GB PCIe，SM80，driver 595.91.07，torch 2.8.0+cu128。
- 固定 step400，layer 0 的七个 Linear；M=1，BF16 输入/输出、FP32 累积。
- 权重是真实已验收 payload，输入是固定种子合成激活，不是采集的模型轨迹。
- baseline 是同一 step400 解码后的 dense BF16 matmul，不是原 BF16 模型参数。
- 源码 `2868d5fa9763bfaa90ea1db786dea8a4c923783e`；groups_per_split=8。
- warmup 20、每轮 100 次、7 轮交替计时；相对极差不超过 10% 才报告该格速度比。
- 转换/JIT/加载不计时；包含两个 kernel launch、FP32 scratch 归并；输出预分配。

## CUDA Graph 同轮结果

速度比定义为 dense 时间 / packed 时间，小于 1 表示变慢。单位 μs。

| Linear | [O,K] | Dense BF16 | Packed v1 | 速度比 |
|---|---|---:|---:|---:|
| self_attn.q_proj | 4096 × 4096 | 22.424 | 50.248 | 0.4463× |
| self_attn.k_proj | 1024 × 4096 | 8.037 | 18.186 | 0.4420× |
| self_attn.v_proj | 1024 × 4096 | 8.247 | 18.133 | 0.4548× |
| self_attn.o_proj | 4096 × 4096 | 23.024 | 50.260 | 0.4581× |
| mlp.gate_proj | 12288 × 4096 | 64.069 | 134.043 | 0.4780× |
| mlp.up_proj | 12288 × 4096 | 63.964 | 135.539 | 0.4719× |
| mlp.down_proj | 4096 × 12288 | 67.034 | 136.132 | 0.4924× |

这七项均通过数值、重复执行和计时稳定性检查。三个随机输入及一个交替符号输入
用于数值检查；Graph capture/replay 另有数值检查。不同 module 单独报告，不能
把同形状结果视为完全相同权重。结果表不能推导完整 block 的加速比。

## Eager 结果与验收

Eager 七项数值均通过。六项稳定结果速度比为 0.4405–0.4989×；k_proj 的 dense
baseline 相对极差为 31.29%，超过 10% 门槛，因此该格速度比不接受，整轮为
`completed_unstable`，没有通过修改阈值或删除样本掩盖波动。

Graph 轮为 `accepted_execution_negative_speed`；Eager 为
`accepted_correctness_records_timing_unstable`。两项作业退出码均为 0。
审核重算计时统计，核对源码/environment/manifest/payload 绑定，重新转换并核对
七个 module 的源及目标 tensor hashes，检查每格四项记录的数值门槛与重复一致性。
审核没有独立重跑前向或保留所有输出向量，数值结论基于 runner 的 gate 记录。

Eager 与 Graph 都明显变慢，说明仅消除调用开销不足以解决差距；这不等于已完成
kernel 内部瓶颈归因。尚未运行 profiler，不能断言是访存、同步或算术中的哪项主导。

## 证据与下一步

- Eager result SHA256：`8eb643383ed8ab1d013c3268000f45afaf7179fb86cbe464fe11d517d9d980bb`
- Graph result SHA256：`ace19fb0f7f634188e8f55fd7d2b5bd77844f220b1dbf2bccadf2c08c78162cb`
- 远端结果及 acceptance：`/workspace/results/m1-a100-20260914/linear-layer0-{eager,graph}-v1*.json`
- 远端作业与日志：`/workspace/jobs/m1-linear-layer0-{eager,graph}-v1/`
- 本地私有备份：`server_results/runpod_m1_a100_2026-09-14/linear-{eager,graph}-v1/`

下一步先做有界的 M=1 kernel 瓶颈定位及优化，保留 v1 数值/速度基线与冻结权重；
不因该轮 kernel 变慢直接推进 block/full-model，也不修改量化语义或数值容限。
本轮没有启动 block、full-model、vLLM 或新的精度实验。

## A100 SXM4 候选批次与 block（2026-09-14）

新服务器 A100 SXM4 80GB，driver 580.126.20，torch 2.8.0+cu128；与上次 PCIe
结果分开。源码 `6ba5076b1d40b20aec709fed9d3cdd0de97c5b12`，原网络卷与固定 step400。
统一 trial OMP/MKL/torch CPU 线程为 1，配置已记录在新 environment.json。

6 组固定配置 × eager/Graph × 7 Linear：84 格数值检查通过，77 格计时稳定。
批次运行约 91 秒，未放宽容差；所有稳定格的 packed 都慢于同轮 decoded dense。
按七项 packed median 之和，从所有格均稳定的 Graph trial 中选择 rows4（v2，
groups_per_split=8）进入 block；该和不是 block latency，也不是统计显著最优证明。

| rows4 Graph | dense μs | packed μs | dense/packed |
|---|---:|---:|---:|
| q_proj | 24.055 | 62.995 | 0.382 |
| k_proj | 8.224 | 24.189 | 0.340 |
| v_proj | 8.485 | 24.194 | 0.351 |
| o_proj | 22.444 | 50.465 | 0.445 |
| gate_proj | 64.484 | 116.396 | 0.554 |
| up_proj | 64.460 | 116.990 | 0.551 |
| down_proj | 64.969 | 117.646 | 0.552 |

单 block（layer 0，synthetic hidden，sequence=1、空 KV、eager）：三次数值与
repeat-exact 检查通过，七个 packed Linear 覆盖通过，三条路径七轮计时稳定。
original BF16 1044.932 μs；decoded step400 BF16 1049.944 μs；packed 1213.783 μs。
packed 相对 decoded 的 speedup 为 0.8650×，即 latency 高 15.6%。

用户明确要求不以 block 的速度结果阻止完整模型实验。完整模型 v2/gps8 已于
13:00:15 UTC 独立启动（tmux `m1-sxm4-full`），最后观察 Python PID 4562 运行中；
尚未验收，不能外推速度。输出 `/workspace/results/m1-sxm4-20260914-v1/full-model-rows4.json`。

证据备份：`server_results/runpod_m1_sxm4_2026-09-14/`（private，Git-ignored）。
本地审计重算计时与检查来源 hash，未在审计时独立重跑 forward。

- environment SHA256：`beb77aaae97cd276935f1a3d8aec713679adc70ff034079e86b53332f8df2bda`
- batch SHA256：`53398c5e59db588cb69067e64e18fecc67aa598aefa11846a4e663e3b2096d9e`
- block SHA256：`ac0b458351411869f13a2076e8d5a8aa4c472e6dd796d1a3e0f56a613f2e6bd9`

环境恢复：依赖阶段约 54 秒，首次四 variant 编译/smoke 221.29 秒；全套 85 测试
162.74 秒，无跳过。观察到 ninja 文件系统等待与合成 CPU 小模型高线程开销，
但没有 profiler 定量归因。限制 CPU 线程后专项测试/新记录阶段约 25 秒，此时已有
编译缓存，不能把全部差异归因于线程设置。ptxas 显示三个 v2 行复用版本主 kernel
分别使用 40/44/56 registers、0 spill；这不等于已经测得 occupancy 或主要瓶颈。
