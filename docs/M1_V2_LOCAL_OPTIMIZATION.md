# M=1 v1 差距分析与本地 v2 候选

2026-09-14，仅本地修改。没有连接服务器，没有 CUDA 编译/计时结果。
保留已接受的 v1 负面基线，不把候选设计推断写成性能结论。

## 为什么比 dense 慢一倍以上

已测事实：Graph 下 q_proj 为 dense 22.424 μs / packed 50.248 μs；七项 packed
耗时均为 dense 的 2.03–2.26 倍。Graph 没有消除差距，因此调用开销不是充分解释。

代码解释分为已知设计特征与待 profiler 定量的问题：

1. **v1 是保守的 SIMT 正确性原型，不是 Marlin 的高吞吐实现。**
   本地 Marlin 的 `cp_async*`、`mma`、多阶段流水、shared-memory swizzle、
   条带调度等没有在 v1 中实现。参考其分块思路不等于已复现其性能。
   dense 直接消费 BF16 权重；v1 还需逐元素解码、两组 FP32 乘法/求和、查稀疏
   索引、加入 refinement、舍入 BF16，随后才点积。权重字节更少不保证耗时更低。
   本次没有 profiler 记录，也没有确认 dense baseline 内部使用的具体 kernel；
   不能以“dense 使用 Tensor Core”作为已经证实的归因。
2. **v1 的 FP32 shared-memory scalar 地址模式有四路 bank conflict。**
   对固定 j，warp 读取 `sx[4*lane+j]`、`c0[4*lane+j]`、`c1[4*lane+j]`；
   bank = index mod 32，仅覆盖 8 个 bank，每个 bank 对应四个不同地址。
   这是源码层地址分析；实际 SASS 指令及 stall 占比仍需测量。
   bank 规则见 [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)。
3. **每 CTA 只计算四行，公共数据复用不足。**
   每组加载输入、column scales 和选列映射，经历两次 CTA barrier，只服务四个
   输出行。大量小 CTA 重复执行这些工作；片外实际读流量受缓存影响，不能把重复
   load 指令数直接当成 HBM 流量。
4. **还没有流水隐藏访存、同步与解码延迟。**
   每组是加载 → barrier → 解码/累积 → barrier。是否主要受内存、算术、同步
   或 occupancy 限制，目前未知，需要有界 profiler capture 才能量化。

因此，两倍差距并不证明量化格式无法加速，而是当前实现付出的解码/调度代价
没有被紧凑表示带来的收益抵消。我之前提供的是保守原型，距离 Marlin 级别的优化
还有明确差距。

## v2 的两项有针对性的修改

新文件 `src/fluxbin_style/csrc/m1_v2.cu`；v1 的 `m1.cu` 字节/hash 保持不变。

**共享内存 swizzle**：

```text
shared_index(k) = (k & 3)*32 + ((k >> 2) XOR ((k & 3)*8))
```

它是 0..127 的双射。对于写入的连续 k 和读取的 `4*lane+j`，每 warp 的
32 个 scalar 地址都覆盖 32 个不同 bank。不是简单 transpose：简单 transpose
可能只把读取冲突转移到写入。lookup 在 shared memory 中改用 int32，使 bank
映射与 FP32 数组一致；外部 payload 仍为 int16，没有改变文件格式。

**每 warp 四个输出行，每 CTA 十六行**：

输入/column scales/选列判定在寄存器中复用于四个输出行。保持原来的
`group -> j -> FP32 FMA -> warp reduction -> split reduction` 次序。
默认 split 粒度为八组，工作区和第二个归并 kernel 不变。

| 设计量 | v1 | v2 |
|---|---:|---:|
| CTA threads | 128 | 128 |
| 每 warp 输出行 | 1 | 4 |
| 每 CTA 输出行 | 4 | 16 |
| `[4096,4096]`、groups_per_split=8 的 CTA 数 | 4096 | 1024 |
| 相同 O/G 下公共数据 staging 的 CTA 次数 | 1× | 大尺寸整 tile 时 1/4× |

不承诺 4× 加速：逐权重计算量并未减少四倍，register pressure 还可能降低
occupancy 或导致 spilling。v2 仍没有 MMA/cp.async，多行复用与 swizzle 是这次的
有界改动，性能上限与后续优化方向由实测决定。

## 不改变的合同

- g128/s8、全部符号/索引/FP32 scales 和 versioned external layout 不变。
- global/refinement 在 FP32 合并，然后 BF16/FP16 舍入，再进行点积。
- 保留 FP32 运算顺序、--fmad=false、数值容差与稳定性门槛。
- 不能为了省运算将 row scales 提到点积外：那会改变现有舍入语义。
- v1 默认路径、已保存结果不变；v2 通过显式 `--kernel v2` 选择。
- engine-neutral `M1Backend(kernel='v2')` 可选择候选，没有接入 vLLM。
- block/full-model runner 显式继承所选 kernel 与 split，不悄悄运行默认 v1。

## 下次一次开机完成的有界验证

先在本地提交/推送。源码已变化，旧环境记录中的 source hashes 不再匹配：
下一次在相同软件环境另存新的环境/source 记录，不覆盖旧验收记录。

1. 构建 v2，查看 ptxas 的 register/spill 信息，运行针对性 CUDA 测试。
   新增测试覆盖 O=1/7/16/17/31/33、K split 尾部、FP16/BF16、Graph、非默认
   stream、污染后的 workspace 覆写，并要求 v2 与 v1 逐元素完全一致。
2. layer 0 七个真实 Linear：每个数值输入同时核对 dense oracle 和 v1。
   测试入口新增同轮 dense/v1/v2 计时；任何 v1/v2 不一致都中止，不放宽门槛。
3. 后续已准备固定 6 组（含基线）eager/Graph 批次，见
   [M1_CANDIDATES_FULL_MODEL_RUNBOOK.md](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)；不自动开展无界参数搜索。
4. 若仍明显较慢，下一项有界工作是 profiler：检查 shared bank conflict、
   barrier stall、register spill、occupancy、memory/compute utilization。
   当前没有完成该归因，不根据猜测继续堆优化。

示例（下次服务器运行，当前不执行）：

```bash
python scripts/run_m1_linear_benchmark.py \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --kernel v2 --mode graph --groups-per-split 8 \
  --output "$FLUXBIN_RUN_ROOT/linear-layer0-v2-graph.json"
```

输出 `speedup` 仍是 dense/候选；另加 `speedup_vs_v1`，并记录 v1 同轮计时。
增加 v1 对照后，三项计时全部稳定才报告速度比；保留每轮原始样本。

## kernel v3：Marlin 流水与 MMA 适配

2026-09-15 更新：下述待验证描述属于本地准备阶段。现已在 A100 PCIe 编译并通过
全部 91 项测试；v3 gps4 比 v1/v2 快但仍慢于 dense，全模型未实现加速，第二个
prompt 的辅助基线计时不稳定。详见 [GPU 结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。

这里的 v3 指 deployment kernel，与历史 32B 算法 artifact 的 v3 无关。
新增 `src/fluxbin_style/csrc/m1_v3.cu`，参考本地 Marlin revision
`1f25790bdd49fba53106164a24666dade68d7c90` 的 async copy、MMA、寄存器双缓冲
与分工思路。保留 Apache-2.0 版权说明及 `licenses/Marlin-LICENSE`，打包时附带。

| 机制 | 本次实际实现 |
|---|---|
| global → shared 流水 | 三个 stage，预取两组；`cp.async.commit_group` / `wait_group` |
| 紧凑权重读取 | 每线程搬运 16B codes，row scales 以 8B async copy 读取 |
| shared 布局 | codes 每行 48B pitch，适配 MMA 行组的广播访问，避免 32B pitch 的 bank 冲突 |
| 寄存器解码 | 合并 global + sparse 后生成成对 BF16/FP16 MMA fragment，row scales 每 group 预加载 |
| 解码与矩阵计算 | 两份 fragment 双缓冲，预备下一 K16 fragment，再调用当前 `mma.sync` |
| Tensor Core | `m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32` |
| CTA 分工 | 128 threads、64 输出行，受 SM 数量约束的 persistent tile/split 任务遍历 |
| 归并 | split>1 保留固定次序 FP32 scratch reduction；单 split 直接写输出，省掉第二次 launch |

MMA operand A 放 16 行权重，B 放激活，激活复制到 8 个列位置，最后只写 column 0。
M=1 因而仍有 8 列中的冗余计算，是否能被 Tensor Core 吞吐/流水优势抵消，必须实测。
没有照搬 Marlin 的跨 CTA 自旋锁归并；当前固定 scratch 归并避免引入调度等待。
2B sparse code 因奇数 O 时的对齐约束使用标量预取，其余 bulk 数据异步搬运。
输入对齐由 host 显式检查（主体 16B、row scales 8B），不接受不满足 async-copy 对齐
的 storage-offset 输入。标准转换产物和新分配输入满足该要求。

当前离线 `[G,O,...]` layout 和算法权重不变，未改为 INT4，也没有复制 dense 权重。
权重合并后 BF16/FP16 舍入保留；点积顺序改为 MMA。v1/v2 源码不改、默认路径不改；
通过 `--kernel v3` 或 `M1Backend(kernel='v3')` 显式选择，block/full-model 自动继承。
v3 同轮记录 v1 输出差异与计时，但不要求 MMA 与 SIMT 逐位一致；dense numerical gate、
重复输出和路径检查仍保留。全模型可继续使用用户指定的 report-only 数值策略。

索引依据：[NVIDIA PTX 的 m16n8k16 fragment 映射](https://docs.nvidia.com/cuda/archive/11.6.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。
CPU 索引模型已核对 16×16 fragment 的完整覆盖、combined-weight 重建、输出列选择、
shared pitch、三段槽位复用和 persistent task 完整性；这些模型测试不能替代 GPU 执行。
CUDA 测试入口新增 O=1/7/17/63/64/65/129/65535、group=1/2/3/4/9/10、
gps=1/2/8/1024、BF16/FP16、one-hot/random、workspace 污染、非默认 stream 和 Graph。
大 O case 用于覆盖持久 CTA 的多任务执行。**当前没有 NVCC 编译、CUDA 数值或性能结果。**

本地验证：91 项测试，88 通过、3 项 CUDA 测试跳过；wheel 打包通过，并检查包含
`m1_v3.cu` 和 Marlin 许可证。wheel 构建不编译 CUDA 扩展，不能视为 NVCC 验证。

## 下一步：外提 scale，避免逐权重浮点重建

2026-09-15 用户取消诊断实验，提出以上方向；本节记录候选设计，尚未实现或
运行新 kernel，也未修改旧 oracle。已提交的孤立 Linear profiler 保留但暂停，
全模型 profiler 准备未实施。持久环境复用入口继续保留。

### 实测与静态推断分开

同轮 v3/gps4 Graph q_proj [4096,4096] 的 dense/packed 时间为
23.019519 / 43.120642 µs，速度比是 **0.53384×**，不是 0.43×。
主要 codes 与 row scales 合计 3.125 bit/weight；完整转换 layout 实际为
6,597,120 bytes，还包括 column scales、indices 与 lookup。
用完整 layout 字节数除以 packed 时间得到约 153.0 GB/s；它只是按唯一存储字节
计算的有效速率，不是实测 DRAM throughput。CTA 重复读取、L2 命中、激活和
workspace 流量尚未由计数器测量。

按 dense 唯一权重字节/23.019519 µs 得到约 1457.7 GB/s，再据此折算 packed
约 4.53 µs，只能当作理想参考。跨 GH200/A100、不同形状和缓存状态的 roofline
达成率不能直接等价；QBB A8/BMMA 实测也不能用另一 SIMT sign-add kernel 的
静态指令数解释。未在这里重新验收用户提供的 QBB 1.82× 那个具体实验。

代码确实逐权重进行符号提取、row/column 乘法、sparse lookup 和类型转换。
但“13 条指令 + 16 B shared/weight”、ALU 11 µs、LDS 14 µs、总下限 15–25 µs
均是尚未验证的静态估算，不是性能上限结论：编译器复用、广播、实际 SASS、
流水重叠和依赖链会改变结果。剩余耗时不能全部归因于 occupancy/barrier。
运算与 scale 流量也会随 base 数变化，不能称为完全与 base 数无关。

### 候选计算方式

每个 group 中 global column scales 不含输出行维，因此可复用：

```text
z[b,k] = FP32(x[k]) * col[g,k,b]
global[o] = sum_b row[g,o,b] * sum_k sign[g,o,k,b] * z[b,k]
sparse[o] = sum_b sparse_row[g,o,b] *
            sum_{j=0..7} sparse_sign[g,o,j,b] *
                         (FP32(x[index[g,j]]) * sparse_col[g,j,b])
y[o] = sum_g (global[o] + sparse[o])
```

先考虑两路 activation 变换复用与 8 个 sparse 位置单独计算，再选择 sign-add、
LUT 或其他适合该结构的实现；不把普通 FP16 MMA 当作必须保留的约束。
不重新量化，不改 payload。该重排在实数运算下等价，但与旧的
“合并权重先舍入 BF16 再点积”并非同一浮点程序。

### 新参考的边界

新候选应独立标明 reference/version，保留旧 dense-BF16 重建参照和历史报告。
将固定 FP32 payload 系数提升到 FP64 后计算结构权重及点积，可作为高精度诊断
参照；候选 FP32 累积另行规定容差。不能把 FP32 本身称为“精确重建”，也不能
保证改变舍入次序就必然更接近原始 BF16 模型或改善全模型误差/PPL。
去掉权重 BF16 舍入减少一种误差来源，但乘加、归并和激活变换仍会舍入。

v3 已经不要求与 v1 逐位一致，只要求算子 dense 数值门槛和自身重复一致性；
全模型采用 report-only。旧全模型不通过数值门槛不能证明换 oracle 后会通过。
后续同时记录候选对结构高精度参照、旧 dense-BF16 参照的误差；完整模型主速度
基线继续为 original BF16。新参照的最终容差与实现验收需随候选明确记录，
不通过覆盖或改写旧结果来获得“通过”。


## v4 已实现：全局激活变换 + 符号点积（本地，待 GPU）

新增 `m1_v4.cu`，显式选择 `kernel='v4'`；v1–v3 源码和默认 kernel 不变。
不再重建逐元素 BF16 权重，不用 MMA，不重新量化。运算合同为 `factored_fp32_v1`：

1. `prepare_x` 每 group 一次，计算 256 个 global x×column FP32 乘积及
   16 个 sparse 乘积；使用 8 个已验证 indices，供所有输出行复用。
2. `factored_m1` 每 CTA 16 行、128 threads，每 warp 4 行；读取 packed signs，
   用符号位操作和 FP32 加法求两路点积。warp 归并后由 lane0 用 FP32 FMA
   乘 row scale 并累加，sparse 只遍历 8 个选中位置，不逐 K 查询 lookup。
3. 固定次序 FP32 split 归并，最后一次转换 BF16/FP16。即使单 split，目前仍
   保留 finish kernel；总共三个 launch，全部计入性能，不隐藏变换和归并成本。

变换缓冲区按 lane 读取次序排列；共享内存 272 个 float，global 每组两次 CTA
barrier。它仍有变换 launch、shared staging、shuffle reduction、split scratch
等成本，不能据静态代码宣称超过 BF16 或已经消除瓶颈。

`workspace_shape(O,K,gps,kernel='v4')` 返回一维 FP32 缓冲区：
`G*272 + ceil(G/gps)*O` 个元素。前部存变换激活，后部存 partial；调用者预分配，
每次全部覆写，无算子内分配。旧 kernel 的二维 workspace ABI 保留。HF adapter
和保留的 engine 接口均按 kernel 分配，vLLM 仍未接入。

新增 `factored_reference.py` 的 `hybrid-structural-fp64-v1`：将现有 FP32 payload
系数提升 FP64，逐 group 重建并与输入做 FP64 点积，不做权重 BF16 舍入。
这是高精度参照而非实数精确解。新 Linear gate 保留输出 dtype 容差：
BF16 elementwise factor .02 / NRMSE .005；FP16 .002 / .0005。
同时记录旧 dense BF16 误差与 v1 输出差异，它们不作为 v4 Linear 主数值 gate。
计时 baseline 仍是旧 decoded BF16 matmul，高精度参照不进入计时。

block 保留旧 dense BF16 兼容性检查；完整模型继续用旧 decoded BF16 记录数值
差异、original BF16 作为主性能基线，允许显式 report-only。prefill 的 dense
fallback 仍采用原先舍入权重，不能称为整条模型已切到结构 FP64 参照。

新候选配置 `configs/acceleration/m1_factored_candidates_v1.json`：v3/gps4 对照、
v4 gps1/2/4/8/16，eager/Graph 共 12 trial。按原 runbook 的 --config 入口运行。
源码已变，上机必须新建 source-matched environment 并通过 v4 CUDA 测试，旧
环境/Linear/block 记录不能直接作为新源码的晋级凭据。当前没有 CUDA 编译、
性能或全模型 v4 结果；诊断 profiler 仍暂停。

本地验证：98 项测试，94 通过、4 项 CUDA 跳过；CLI 参数检查、wheel 打包和
源码/参照打包检查通过。wheel 不编译 CUDA，不能代替上机验收。

2026-09-15 GPU 更新：v4 已编译，98 项测试通过。外提版实测仍慢于 v3/dense，
完整模型速度为 original BF16 的 0.83063× / 0.83143×；详见
[完整结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。此前待验证描述属于本地实现阶段。


## v5 LUT-A16：2026-09-15 本地实现

`src/fluxbin_style/csrc/m1_v5.cu` 参考 QBB 的 eight-sign LUT 和线程持有输出行的
计算方式；未引入 A8。v4 的 FP32 分解内积合同及结构 FP64 参照继续使用，
但 LUT 递推改变 FP32 求和顺序，不要求与 v4 bit-match，也不宣称精确实数运算。
原始 BF16 仍是全模型性能主基线；旧 decoded BF16 数值误差单独保留。

- 256 threads / CTA，1024 输出行 tile，每线程独立处理 4 行。
- 两个 base 的 column scales 不同，分别建 16 张 256-entry FP32 表；
  sparse 两个 base 各一张表。共 34,816 B shared memory。
- 每张表由一个 warp 建立：每 lane 先计算一个五位 seed 对应的完整八项和，
  再通过上三位的增量递推生成其余条目。读取依赖均在同一 lane 内。
- 每 8 个权重提取两个符号 pattern，分别查表；每行每组 32 次 global-base
  查表、2 次 sparse 查表，然后 4 个 row-scale FMA。不再逐权重重建或 warp 归约。
- 建表与计算融合；每组建表后、读取结束后各一次 CTA barrier。多 split 两次
  launch（主计算 + 固定顺序 FP32 reduction），单 split 一次，最终才舍入输出。
- 保持原 packed payload/layout，正常转换生成的 code buffers 满足向量读取对齐；
  ABI 拒绝不满足 codes 16-byte / sparse 2-byte 对齐的外部视图。
  workspace 为 `[ceil(G/gps), O]` FP32，所有位置每次覆写，无内部动态分配。

这版仍有建表、shared 查表冲突以及小形状下 CTA 数不足的潜在成本，尚未测量。
1024 行 tile 和固定 split 候选只是待验证实现，不能据此承诺超过 BF16。

本地在 macOS arm64 用实际 `lut8.cuh` 编译 C++ 检查：穷举 65,536 个双 base
交织符号编码；100 组输入的全部 256 个 LUT 项与独立 FP64 sign-dot 对照通过。
完整测试 102 项：97 通过、5 CUDA 跳过。新 CUDA 测试覆盖 FP16/BF16、one-hot/
随机输入、1024 行边界、K split 尾部、NaN workspace 覆写、非默认 stream 与 Graph。
CUDA 测试已准备但未执行；没有连接服务器或生成新的性能证据。
新增 `.cuh` 纳入 wheel 和环境/Linear/block/full-model 源码 hash 链。


## 延后归约与离线 byte planes：2026-09-15

`v4_late` 是单独的 `m1_v4_late.cu`，保留原 v4 作为同轮对照。每 lane 对其
局部 global/sparse partial 应用相应组的 row scales，再累加所有组；每行每个
K split 最后只做一次 warp_sum。不是全 K 必然只归约一次：仍有独立 split
结果和最终 reduction。组长为 gps 时，完整 split 的 warp_sum 次数由 4*gps
降为 1；所有 lane 做 row-scale FMA，FP32 加法次序改变，沿用结构参照与原容差，
不要求 v4 bit-match。激活变换及三次 launch 仍计时。

`v5_p256/p512/p1024` 是 v5 的编译期专化，分别使用 256/512/1024 行 tile。
`convert_artifact(..., kernel=...)` 离线将 global codes 转为 `[G,O,2,16]`，
sparse codes 转为 `[G,O,2,1]`；每个 byte 已是一个 base 的八个 sign bits。
符号位数量不变，反转换逐位无损，既不增加一份常驻原符号布局，也不改 scales。
使用独立 format `fluxbin-hybrid-g128-s8-m1-planar-v1`；模型适配器与预留的
engine interface 都通过所选 kernel 转换。旧布局仍用于旧 kernel。

LUT 内层直接取 byte 索引，不再调用 pattern()；建表、34,816 B shared、FP32
累加及 split reduction 与 v5 相同。较小 tile 增加 CTA 数但重复建表更多，固定
候选比较该取舍，不提前选胜者。Linear benchmark 的 v1 诊断单独使用旧布局，
不会把 planar 数据传给旧 kernel；原始 BF16 全模型主性能基线不变。

本地完整测试 105 项：100 通过、5 CUDA 跳过。新增离线布局穷举 65,536 种编码、
还原、FP64 结构参照一致性、字节数相等、后端和模型预填充适配测试。
CUDA tests 已包含 v4_late、三种 planar tile、tail/stream/Graph/FP16/BF16，
尚未执行，不宣称 NVCC 编译或加速通过。

静态分析边界：payload bytes/time 是有效带宽，不是 DRAM transaction 实测。
CUDA 源操作数不能直接等同 SASS 指令数；发射、shuffle、依赖延迟存在重叠，
不能简单相加来证明 45 us 的耗时归因。以上修改针对已确认的重复工作，
不把静态估算当作经过 profiler 验证的瓶颈比例或硬性能上限。
