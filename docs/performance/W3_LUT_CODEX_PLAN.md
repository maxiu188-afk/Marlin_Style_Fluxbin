# Codex Task Plan — GPTQ W3 Planar LUT Kernel (M=1)

## 目标

实现一个 **GPTQ W3、group=128、M=1 decode-only** 的 CUDA LUT kernel，作为当前 v5 / FluxBin-style kernel 的直接对照基线。

核心目标不是做最终 W3 backend，而是回答：

1. W3 在 M=1 下用 bit-plane LUT 能跑多快？
2. 相比 decoded BF16 和当前 v5，瓶颈在哪里？
3. 更小 shared footprint 是否能改善小输出维度（尤其 k/v）的 CTA 并发问题？

**本轮不做 batch>1、不做 Tensor Core/Marlin W3、不做 QKV fusion、不改 finish_m1、
不实现 prepare candidate。第一轮只验收 inline W3 LUT。**

---

## 1. 量化与数学形式

GPTQ W3：

- `bits = 3`
- `group_size = 128`
- `q ∈ {0,...,7}`
- `w = scale * (q - zero)`

将每个 3-bit code 拆为：

```text
q = b0 + 2*b1 + 4*b2
```

每 8 个 activation 建一个 unsigned LUT：

```text
table[c][p] = Σ_{j=0..7} bit_j(p) * x[8*c+j]
```

一个 group 有 16 个 chunk，因此只需：

```text
16 × 256 float LUT
```

三个位平面共享同一套 LUT。

### constant-zero fast path

先通过现有 GPTQ decoder **解码后**检查真实 zero-point。

若所有 zero 均为 4，则 converter 离线翻转最高位：

```text
plane0 = bit0(q)
plane1 = bit1(q)
plane2 = NOT bit2(q)
```

此时：

```text
q - 4 = b0 + 2*b1 - 4*b2
```

runtime 直接：

```text
T = a0 + 2*a1 - 4*a2
y += scale * T
```

不需要 `Sx`、zero metadata 或额外 zero correction。

若 decoded zero 不是恒 4，再实现 generic path；不要提前复杂化。

---

## 2. 离线格式

不要直接使用 GPTQ 原生 3-bit packed layout。

统一转换为：

```text
planes[G][3][O][16]   uint8
scales[G][O]          bf16
perm[K]               int16
```

其中：

```text
G = K / 128
```

采用 **plane-major**：

```text
[G][plane][O][16]
```

这样每线程每 plane 可用一次 16B load，warp 内连续 output row 完全合并。

### g_idx

先严格遵循 GPTQModel 7.4.0 canonical semantics：raw `FORMAT.GPTQ` 使用
`10-1-10-1-10` 三个 int32 表示 32 个 W3 code，raw qzero 是 format-v1；
Torch backend 加载时会把 qzero 转为 format-v2。converter 必须显式声明读取的是
raw v1 还是 loaded v2，禁止猜测。

`desc_act=True` 时，`g_idx[k]` 表示原始输入列 `k` 使用的量化 group。weight 是
静态的，因此必须按每个 Linear 独立离线规范化：

```text
perm = stable_argsort(g_idx)
q_sorted[s, o] = q_original[perm[s], o]
x_sorted[s]    = x_original[perm[s]]
```

converter 必须先按 canonical W3 规则解出整数 code，再沿 K 维按 `perm` 重排，
使 kernel 永远看到连续 group128。验收必须确认 `g_idx[perm]` 严格等于
`repeat_interleave(arange(G), 128)`，且 `perm` 是 `0..K-1` 的双射。

若 permutation 非 identity：

- 保存 `perm[K]`
- runtime activation 也必须按相同顺序读取
- v1 在 inline LUT build 阶段直接按 `perm` gather `x`
- 另保留 materialized `x_perm` reference path 做正确性与性能对照

**禁止在 main kernel 中按 irregular g_idx 访问 weight。**

---

## 3. 先做 checkpoint inspection

CUDA 前先实现并打印：

```text
bits
group_size
qweight shape/dtype
qzeros shape/dtype
scales shape/dtype
g_idx shape/dtype
desc_act
decoded zero unique values
g_idx 是否等价于连续 group
```

重点确认：

1. raw qzero v1 与 loaded qzero v2 的解释是否一致
2. decoded zero 是否在全部 252 Linear 上恒为 4
3. 每个 Linear 的 `perm` 是否 identity；禁止从一个 Linear 外推到其他 Linear
4. 同向重排 weight/input 是否与原始 `g_idx` matvec 完全一致

---

## 4. Converter 与 round-trip

新增：

```text
src/fluxbin_style/gptq_deployment.py
```

至少实现：

```python
inspect_gptq_checkpoint(...)
convert_gptq_w3_to_planar(...)
restore_planar_w3(...)
```

如需要：

```python
build_input_permutation(...)
```

### 验收

已有 decoded GPTQ BF16 layer 作为 oracle：

```text
packed GPTQ
 -> existing decoder
 -> BF16 reference

packed GPTQ
 -> new planar converter
 -> CPU restore
 -> BF16 rounding
```

要求按顺序通过：

```text
integer q codes exact
decoded zero exact
raw FP16 scales exact as source provenance
deployment BF16 scales exact after GPTQModel loaded-dtype cast
BF16 bitwise identical
```

优先对现有全部 decoded layer 跑 round-trip。

---

## 5. Unsigned LUT builder

新增：

```text
src/fluxbin_style/csrc/lut8_unsigned.cuh
```

不要修改现有 `lut8.cuh`。

从现有 `build_lane` 派生：

```cpp
build_lane_unsigned(...)
```

区别：

```text
signed binary: bit ? +v : -v
W3 unsigned:   bit ?  v :  0
```

目标 shared：

```text
float tbl[16][256]
```

总 shared：

```text
16 KB
```

---

## 6. CUDA kernel

新增：

```text
src/fluxbin_style/csrc/w3_lut.cu
```

只支持：

```text
M = 1
group = 128
W3
```

尽量复用当前 v5：

- CTA / row ownership
- split-K
- finish_m1
- workspace ABI
- benchmark infrastructure

不要顺手重构已有实现。

### main loop 逻辑

```cpp
for each group g:
    build unsigned LUT for x[g*128 : g*128+128]
    __syncthreads()

    for each owned output row o:
        load uint4 plane0
        load uint4 plane1
        load uint4 plane2

        a0 = a1 = a2 = 0
        for c in 0..15:
            a0 += tbl[c][byte(P0,c)]
            a1 += tbl[c][byte(P1,c)]
            a2 += tbl[c][byte(P2,c)]

        T = a0 + 2*a1 - 4*a2     // zero=4 fast path
        result += scale[g,o] * T

    __syncthreads()
```

v1 可以 W3 专用，不要做 generic bitwidth template。

---

## 7. 第一轮只做 inline build

每个 CTA 自己建 LUT，然后处理自己的 rows。

`prepare_lut[G][16][256]` 不作为第一轮 candidate，也不提前实现。先完成 inline
正确性、12-cell CUDA Graph benchmark，再 profile 各 shape 的最佳 inline candidate。
只有 profiler 证明 LUT construction 是主要瓶颈时，才另开诊断分支实现 prepare。
即使进入该分支，正式比较仍必须使用包含 gather/prepare/main/finish 的 Graph total。

---

## 8. Trial matrix

固定：

```text
M = 1
group = 128
gps = 1
```

变量：

```text
row tile R: 256 / 512 / 1024
shape class:
  - q/o:      self_attn.q_proj   [4096,4096]
  - k/v:      self_attn.k_proj   [1024,4096]
  - gate/up:  mlp.gate_proj      [12288,4096]
  - down:     mlp.down_proj      [4096,12288]
```

总计：

```text
3 × 4 = 12 trials
```

本轮不要加别的维度。

---

## 9. Benchmark baseline

新增：

```text
scripts/run_w3_lut_benchmark.py
configs/acceleration/w3_lut_candidates_v1.json
```

正式主指标固定为 **同一 GPU job、同一输入、同一轮次的 CUDA Graph total**。
Graph capture 必须覆盖 inline LUT build、main compute 和 `finish_m1`；JIT、load 与
离线 conversion 不计时。eager 只用于调试，不进入第一轮正式 summary。

每个 cell 同时跑：

1. **GPTQ decoded BF16** — primary dense baseline
2. 原始 BF16 weight — secondary baseline
3. 当前稳定 `v5_p1024/gps1` — same-run anchor
4. inline W3 LUT candidate

不要和历史旧数字直接比较 speedup，也不要用 prepare/main/finish 的分项相加替代
一次完整 Graph total。

---

## 10. Correctness contract

### converter

要求 BF16 round-trip bitwise identical。

### kernel

不要要求与 decoded-BF16 GEMV bitwise 一致。

W3 LUT 直接计算：

```text
scale * integer_code * x
```

而 decoded baseline 先把 weight 舍入 BF16。

kernel correctness 使用结构 reference：

```text
q + scale + zero/permutation
 -> FP64/FP32 reference matvec
```

同时记录：

```text
error_vs_structural_reference
error_vs_decoded_bf16
```

---

## 11. Profiling

12-cell 正式 latency 完成后，只 profile：

```text
best stable inline candidate per shape class
v5_p1024
```

至少记录：

```text
registers/thread
shared memory/CTA
active CTA/SM
achieved occupancy
eligible warps/scheduler
DRAM throughput
HBM bytes
shared load traffic
shared bank conflicts
global sectors/request
kernel duration
```

重点验证：

### H1
16 KB shared 是否在 grid 足够时提升 CTA/SM residency；small-O `R=1024` 的
grid underfill 必须与 shared-memory occupancy 分开归因。

### H2
48 次 LUT lookup / (group,row) 是否让 kernel 从 HBM-bound 转为 instruction/shared-memory-bound。

不要预设一定能从 4 CTA/SM 提升到 5，以 profiler 为准。

---

## 12. 最终输出

生成 summary table：

```text
shape
candidate
R

correctness
cuda_graph_total_us

decoded_bf16_us
original_bf16_us
v5_anchor_us

speedup_vs_decoded_bf16
speedup_vs_original_bf16
relative_to_v5

regs_per_thread
smem_per_cta
active_cta_per_sm
dram_gbps
```

并回答：

1. W3 LUT M=1 是否快于 decoded BF16？
2. 是否接近或超过当前 v5？
3. q/o 与 k/v 是否呈现不同瓶颈？
4. 16 KB shared 是否改善 small-O grid underfill？
5. 当前限制主要是 HBM、shared LUT、instruction issue 还是 finish kernel？

---

## 13. 文件范围

建议新增：

```text
src/fluxbin_style/csrc/lut8_unsigned.cuh
src/fluxbin_style/csrc/w3_lut.cu
src/fluxbin_style/gptq_deployment.py
src/fluxbin_style/w3_lut_deployment.py
src/fluxbin_style/w3_lut_artifacts.py

scripts/prepare_qwen3_8b_w3_lut_artifacts.py
scripts/run_w3_lut_benchmark.py
configs/acceleration/w3_lut_candidates_v1.json

tests/test_gptq_w3_planar.py
tests/test_w3_lut_correctness.py
```

格式 ID：

```text
gptq-w3-g128-sym-planar-m1-v1
```

---

## 14. 实施顺序

```text
Phase 0
confirm GPTQModel-7.4 canonical W3/qzero semantics
inspect all 252 g_idx / decoded zero / packing semantics

Phase 1
CPU converter + restore
integer-code/zero/scale exact round-trip
BF16 bitwise round-trip

Phase 2
unsigned LUT builder
synthetic correctness

Phase 3
inline W3 kernel
M=1 correctness

Phase 4
CUDA build/synthetic correctness preflight on target GPU

Phase 5
4-shape × 3-row-tile same-run CUDA Graph total benchmark
+ decoded BF16
+ original BF16
+ v5_p1024

Phase 6
profile best stable inline candidate per shape

Phase 7
only if LUT build is the measured primary bottleneck:
authorize a separate prepare diagnostic plan

Phase 8
write inline summary + recommendation
```

---

## 本轮明确禁止

```text
batch > 1
Tensor Core / MARLIN W3
QKV fusion
new split-K design
finish_m1 rewrite
hierarchical W2
generic arbitrary-bit kernel
large framework refactor
prepare LUT implementation before inline profiling evidence
```

本轮目标只有一个：

> **把 W3 在 M=1 下的 LUT 性能上限和真实瓶颈测清楚。**
