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
