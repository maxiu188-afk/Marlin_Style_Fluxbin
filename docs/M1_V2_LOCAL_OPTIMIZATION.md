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
