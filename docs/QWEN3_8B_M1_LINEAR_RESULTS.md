# Qwen3-8B 首轮 M=1 Linear 结果

## 内积实现与现有证据（2026-09-15）

不能直接照搬 Marlin 的便宜 INT4 解包，但可参考流水、布局和调度；v3 实际
使用了 MMA。v4 起改用分解内积，v5 起使用 LUT，不能再描述成 Marlin 内积。

| 实现 | Inner compute | 当前证据 |
|---|---|---|
| v1 | 每权重重建、舍入 BF16，再 SIMT 点积 | PCIe Graph 七项 540.846 us（v3 同轮） |
| v2 | 保留重建，改善 shared 访问及多行复用 | PCIe 同轮 501.606 us；另轮 SXM4 全模型 0.9332x/0.9562x |
| v3 | 重建后的 BF16 送 MMA，三阶段流水 | PCIe 同轮 401.408 us，dense 258.212；全模型 prompt0 0.86887x，prompt1 未通过整组稳定性 |
| v4 | 列 scale 外提、SIMT sign-add；每行每组四次 warp_sum | SXM4 七项 462.531 us，dense 256.030；同轮 v3 412.883 us；全模型 0.83063x/0.83143x |
| v4_late | 每 lane 先乘 row scale 累加，每行每 split 最后归约一次 | 本地实现，GPU 待验证 |
| v5 | 每线程持有输出行，8-sign LUT；运行时拆交织符号位 | 本地实现，GPU 待验证 |
| v5_p256/p512/p1024 | 离线 byte planes，直接取 LUT 索引；比较三个行 tile | 本地实现，GPU 待验证 |

Linear 数字为七个独立 Graph Linear 耗时之和，不是 block 延迟；不能跨 GPU
作因果比较。全模型速度比为 original BF16 / packed，均是 report-only 数值
策略下的计时结果，不意味着旧 dense BF16 数值等价。详细来源与门槛见下文。
新候选的实现、检查和运行配置见 `M1_V2_LOCAL_OPTIMIZATION.md` 与
`M1_CANDIDATES_FULL_MODEL_RUNBOOK.md`。

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

### 完整模型首轮失败更新

完整模型任务退出 1，异常为 `full-model correctness/coverage failed: packed_step400:0:0`。
这是冻结数值 gate 触发的主动停止；日志未报告 CUDA 异常或 OOM。
原始 BF16 与 decoded step400 两个基线已完成；packed 完成首 prompt 的一次测量后，
因相对 decoded 的数值不合格而中止，不能发布合格的完整模型加速结论。

252 个 Linear 覆盖通过，输入 continuation 完全相同，KV 长度为 43。
33 个 next-token prediction 中 index=2 不同（decoded=3410，packed=1128）。
logits max_abs_error=0.375，NRMSE=0.0115685276（上限 0.005），
max_abs_logprob_error=0.3394165039（上限 0.05）。
单层数值通过不保证堆叠后的全模型数值通过；具体偏差来源尚未定位，不能直接归因于
量化质量或正常累积误差。没有放宽门槛、重跑或修改 kernel。

失败 JSON、日志和退出码备份到
`server_results/runpod_m1_sxm4_2026-09-14/full-model/`，保留原始现场。

### 完整模型性能重跑完成（report-only）

用户明确接受数值差异记录后，以 `--numerical-policy report-only` 重跑；未改变
权重、kernel、输入、数值容差、decode 长度或重复次数。源码 revision
`d1bcd6da4b8fe6afa210eb286de3baf7a445a077`，仅 runner 的停止策略等修改；底层 src
hash 与此前环境/block 一致。退出码 0，任务总耗时 135 秒，最终状态
`completed_with_numerical_differences`。

主要比较对象为原始 BF16 模型。Qwen3-8B、batch=1、HF eager dynamic KV、两个
固定 prompt、各 32 个 decode steps、每 arm/prompt warmup=1/repeats=3。全部
decode wall/device timing 的相对极差满足 <=10%；吞吐为 32 / decode wall time。

| Prompt | 原始 BF16 ms/token | packed ms/token | 原始 BF16 tok/s | packed tok/s | BF16/packed 速度比 |
|---|---:|---:|---:|---:|---:|
| 0 | 36.036 | 38.617 | 27.750 | 25.895 | 0.9332× |
| 1 | 36.326 | 37.989 | 27.528 | 26.323 | 0.9562× |

结论：当前完整模型没有加速，packed latency 分别高 7.16% / 4.58%。这是两个
固定短上下文的测量，不代表所有上下文/serving 工作负载。辅助 decoded-step400
基线分别为 30.014 / 30.277 tok/s，packed 相对该基线为 0.8628× / 0.8694×。

- 原始 BF16 prefill wall median：36.205 / 36.212 ms；packed：781.886 / 780.345 ms。
  packed prefill 当前使用逐层按需 dense 重建，开销计入该阶段。
- 原始 BF16 resident allocated：15.266 GiB；packed：4.924 GiB。运行峰值约
  15.291 / 5.627 GiB。此为 PyTorch allocated memory，非设备总使用量。
- packed 加载/转换峰值仍为 15.343 GiB，因为先加载 dense checkpoint 再替换。
- 所有 packed trace 都有 252 个目标 Linear，各 prefill fallback 1 次、decode
  packed 32 次；输入 continuation 与 decoded 相同，KV 长度通过。
- 两个 prompt 都有 1/33 greedy prediction 不同，三次重复中的差异相同。
  NRMSE 为 0.0115685 / 0.0113527，最大 logprob 差为 0.3394165 / 0.5623779。
  按用户授权仅记录，不据此中止计时，也不标记为数值等价通过。

结果 SHA256：`36869de3ba17acbcb762d66d6f92cbcb393d19c514449f245c04298cbef32347`。
远端结果：`/workspace/results/m1-sxm4-20260914-v1/full-model-rows4-reportonly.json`。
本地原始结果、日志、退出码：`server_results/runpod_m1_sxm4_2026-09-14/full-model-reportonly/`。
审计核对远端/本地 hash、source/runner/protocol/environment/block 绑定、原始计时
median/极差/速度比、context 与 routes；未在审计阶段重跑模型。


## 2026-09-15 v3 MMA / A100 PCIe

源码 `9438bae0fe10b60db74296f5c3bde437d6f02882`，A100 80GB PCIe，driver
580.159.04，Python 3.12.3，torch 2.8.0+cu128，CUDA toolkit 12.8。
本轮 GPU 与上一轮 SXM4 不同，不将跨机器耗时差归因于 kernel。

恢复依赖至 pip check 用时 57 秒，CUDA build/smoke 约 98 秒；完整 91 项测试
全部通过，总准备流程 221 秒。v3 FP16/BF16 主 kernel 均使用 76 registers、
17,472 B shared memory，无 spill。基础镜像仍只知 RunPod 默认模板，digest 未确认。

固定新配置 `configs/acceleration/m1_marlin_candidates_v1.json` 完成 12 trials，
84/84 数值检查通过，67/84 格计时稳定。本地下载后重新校验汇总同样为 67/84。
v3 与 v1 逐位一致仅为诊断；dense 数值检查和 repeat/Graph 检查保留。

| 完整稳定 Graph trial | 七个 packed Linear 耗时之和 (µs) |
|---|---:|
| v1 gps8 | 540.846 |
| v2 gps8 | 501.606 |
| v3 gps4 | 401.408 |
| v3 不分 split | 989.256 |

v3 gps8/gps16 Graph 含不稳定格，不能给出完整稳定速度结论。
选择 v3 gps4 Graph 作为 block/full-model 配置；上述求和不是单 block 延迟。
其同轮 dense BF16 合计 258.212 µs，v3 仍慢于 dense。

| Linear | 同权重 dense BF16 µs | v3 gps4 µs | 同轮 v1 gps4 µs |
|---|---:|---:|---:|
| q_proj | 23.020 | 43.121 | 49.777 |
| k_proj | 8.264 | 20.490 | 17.388 |
| v_proj | 8.294 | 20.347 | 17.316 |
| o_proj | 23.173 | 42.977 | 49.674 |
| gate_proj | 64.041 | 91.668 | 134.953 |
| up_proj | 64.061 | 91.597 | 135.209 |
| down_proj | 67.359 | 91.208 | 136.643 |

layer-0 empty-cache block 的三次数值/重复、七条 packed 路径和七轮计时均通过：
original BF16 619.991 µs、decoded step400 BF16 617.523 µs、packed v3 713.380 µs。
相对 original BF16 延迟增加约 15.1%。按用户要求继续运行完整模型，不由 block
速度决定是否提交；全模型使用 original BF16 主性能基线、report-only 数值策略。

证据保存在服务器 `/workspace/results/m1-pcie-20260915-v3/`，本地私有备份
`server_results/runpod_m1_v3_pcie_2026-09-15/`。原始 JSON/log 不提交 Git。

### 同轮完整模型结果

`m1-v3-full` / `full-model-mma-split4-reportonly.json`，退出码 0，用时 312 秒，
状态 `completed_unstable`。312 秒包含模型预检、加载、转换、warmup 和测量，
不等于纯 decode 用时。固定两个 prompt、32 次 decode、三次测量，未改协议。

| Prompt | original BF16 wall ms / tok/s | packed v3 wall ms / tok/s | 正式速度比 |
|---|---:|---:|---:|
| 0 | 672.206 / 47.604 | 773.655 / 41.362 | 0.86887× |
| 1 | 676.218 / 47.322 | 735.850 / 43.487 | 不报告：辅助 decoded timing 不稳定 |

Prompt 0 的 v3 延迟增加 15.09%，没有全模型加速。Prompt 1 的 original 和 packed
自身计时均稳定，但辅助 decoded BF16 的 wall/device 相对极差为 10.4823% / 10.4868%，
超过固定 10% 门槛；保持 runner 的整组无效标记，不事后修改门槛或选择样本。
其 decoded BF16 三次 wall 样本为 689.175 / 695.152 / 762.043 ms。
本轮未自动重跑不稳定格。

两个 prompt 的 packed 六条 trace 全部通过 252 路覆盖、相同 fed tokens 和 KV
长度检查（分别 43 / 42）。每个 prompt 1/33 predictions 不同、三次 logits hash
完全重复；NRMSE 分别 0.0112171 / 0.0128938，max logprob difference 为
0.411823 / 0.627422。按 report-only 保留，不宣称严格数值等价。
original BF16 / packed 常驻 allocated 显存分别 15.266 / 4.924 GiB。

本地核验 source/environment/block hash 链、计时中位数、候选汇总及路径检查通过。
完整模型 JSON SHA256：
`45a369dc6b89e3afbdf53dd37d925e1bb2f35988ddfdb12cd73e91a2dad0d343`。
所有本轮结果和四个阶段 job 日志已打包下载并校验，archive SHA256：
`36a08564bbb21353ead50a8da9a06d9d47b7567e1e487df957dc39883fca9860`。
末次检查无 GPU compute 进程、无 tmux session；未关闭实例或删除网络卷。


## 2026-09-15 v4 factored / A100 SXM4

本轮 kernel 源码来自 `c94c61f`；准备环境修复后测候选/块的 Git revision 为
`7f0f066`。GPU 为 A100-SXM4-80GB，driver 570.124.06，Python 3.12.3、
torch 2.8.0+cu128。v4 不逐权重舍入 BF16，Linear 以结构 FP64 参照验收，
旧 dense BF16 误差另记；这与 v3 的算子数值语义不同。

CUDA 专项 10 项通过，首次新 cache 编译/专项执行约 369.7 秒；完整 98 项通过。
v4 prepare kernel 18 registers，主 kernel 45 registers / 1,088 B shared，
finish 32 registers；均无 spill。三个 launch 全部计时，没有隐藏激活变换成本。

固定 12 trials 完成，84/84 数值检查通过，80/84 计时稳定；下载后本地汇总复核
同为 80/84。v4 gps4 是完整稳定的 v4 配置中 packed Graph 求和最小者，但
**没有超过同轮 v3，更没有超过 dense BF16**。

| Graph 配置 | Packed 七个 Linear 求和 µs | 同轮 dense 求和 µs | 状态 |
|---|---:|---:|---|
| v3 gps4 | 412.883 | 259.988 | 全部稳定 |
| v4 gps1 | 470.598 | 256.737 | 全部稳定 |
| v4 gps2 | 458.638 | 256.756 | 含不稳定格，不作为完整稳定候选 |
| v4 gps4 | 462.531 | 256.030 | 全部稳定，选定 v4 |
| v4 gps8 | 500.245 | 256.343 | 全部稳定 |
| v4 gps16 | 584.619 | 255.668 | 全部稳定 |

| Linear | v4 gps4 µs | 同轮 dense BF16 µs |
|---|---:|---:|
| q_proj | 48.128 | 23.178 |
| k_proj | 19.185 | 8.555 |
| v_proj | 19.263 | 8.482 |
| o_proj | 43.352 | 20.698 |
| gate_proj | 110.406 | 64.630 |
| up_proj | 110.678 | 64.590 |
| down_proj | 111.518 | 65.898 |

这些是孤立 Linear 的实测，不是全模型归因证据。v4 gps4 的结构 FP64 NRMSE
最大约 0.001729，旧 decoded BF16 NRMSE 最大约 0.002719；都保留原始记录。

block 数值/重复/7 条 packed 路径通过，但计时为 `completed_unstable`：
original BF16 / decoded BF16 / packed v4 中位数 896.012 / 896.616 / 1069.787 µs。
它们只作不稳定测量记录，不给正式 block 速度结论。用户要求无论 block 速度
都提交完整模型；新增显式 `--allow-unstable-block-timing`（默认关闭），保留数值、
路径、来源与有限样本检查，不修改 block JSON。full runner revision `42d971a`，
入口相关 7 项本地测试通过，kernel/src hashes 未改变。完整模型仍独立判断计时
稳定性，以 original BF16 为主基线，数值使用 report-only。

证据位置：服务器 `/workspace/results/m1-sxm4-20260915-v4/`，本地私有目录
`server_results/runpod_m1_v4_sxm4_2026-09-15/`。


### v4 完整模型验收

任务 `m1-v4-full` 退出 0，用时 119 秒，状态
`completed_with_numerical_differences`。服务器执行了 7 项 full-runner 入口测试并
通过。block 的不稳定状态保留；完整模型三条 arm、两个 prompt 的 wall/device
计时全部通过原 10% 稳定性门槛，没有重选样本或放宽全模型 timing gate。

| Prompt | Original BF16 wall ms / tok/s | Packed v4 wall ms / tok/s | Original/packed |
|---|---:|---:|---:|
| 0 | 837.430 / 38.212 | 1008.182 / 31.740 | 0.830634× |
| 1 | 836.138 / 38.271 | 1005.659 / 31.820 | 0.831433× |

v4 相对 original BF16 的延迟增加约 20.39% / 20.27%，没有完整模型加速。
外提 scale 的这一版 SIMT 实现未达成目标；该负面结果也不能单独量化具体 stall
或判定整个分解算法的性能上限。未运行用户已经取消的 profiler。

每个 prompt 三次 trace 均通过 252 条路覆盖、同 fed tokens、KV 长度（43/42）
检查，packed predictions 对 decoded BF16 均为 1/33 不同、三次 logits hash
一致。NRMSE 0.01124996 / 0.01256111，max logprob difference 0.40623474 /
0.53112793；按 report-only 留存，未宣称旧 dense-BF16 数值等价。
original BF16 / packed v4 常驻 allocated 为 15.266 / 4.922 GiB，packed 运行
峰值约 5.660 GiB；载入仍为先 dense 后替换，峰值约 15.343 GiB。

本地重核 source/environment/block hash 链、中位数与稳定性、候选汇总、路径检查。
完整模型 JSON SHA256：
`bee8ac440428b2251e543024a3beafb5a006a0a73ba8ebc850c2909c36e0fb2d`。
包含原始失败安装、恢复、本地环境、测试、候选、block、full 日志的归档已下载
并与服务器 SHA256 对齐：
`bfb17131ea4987cef0ef6f6e4c9e9582aae27ade78bd13c64eacd50919bbbcb4`。
末次检查无 GPU compute 进程或 tmux session；实例未由代理关闭，网络卷保留。
