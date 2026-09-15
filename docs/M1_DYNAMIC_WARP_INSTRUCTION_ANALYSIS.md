# M=1 FluxBin dynamic warp instructions per weight byte

更新：2026-09-15。本文是只读源码分析的整理结果；除新增本文外，没有修改 CUDA
实现或实验配置。

## 1. 结论

固定 `q_proj [O,K] = [4096,4096]`、`M=1`、group size 128、A100 后，源码级
估算得到：

- 当前最低的 `I_w` 是 `v5_p1024`，约 `0.174--0.235 warp-inst/weight-byte`；
- `v5_p512` 为 `0.247--0.327`，略高于 p1024，但 q_proj 实测更快，说明
  occupancy、CTA 数和 LUT 摊销会推翻只按 `I_w` 排序的选择；
- `v4 -> v4_late` 将动态 SHFL 减少 16 倍，即下降 93.75%；按包含
  control/address 的总估算，`I_w` 下降约 30.2%--33.5%；
- `v5 -> v5_p1024` 删除运行时 de-interleave 后，源码字面少
  `1,257,472` 条动态 warp integer sites；配对估算的总 `I_w` 下降约
  38.8%--49.9%；
- A100 rough issue/HBM 分界约为 `0.299 warp-inst/B`。p1024 低于该值，p512
  跨越该值，原 v5 位于边界上方；v3/v4/v4_late 明显 instruction-heavy。

这个 roofline 不是 profiler 实测。它不包含依赖延迟、shared-memory bank conflict、
occupancy、L2 命中、内存 transaction、双发射限制或 kernel launch 成本。

## 2. 固定条件与计数定义

```text
O = 4096
K = 4096
M = 1
group_size = 128
G = K / 128 = 32 groups
W = O * K = 16,777,216 weights
```

版本配置：

```text
v3 / v4 / v4_late: groups_per_split = 4, splits = 8
v5 family:          groups_per_split = 1, splits = 32
```

`warp-inst` 按一个 warp 发出一条指令计为 1，不乘 32 lanes。源码中一个标量表达式
若由整个 warp 同时执行，仍只记一个动态 warp instruction site。

本文的 `weight bytes` 是 kernel 从权重 artifact 请求的数据字节数，只包括随矩阵规模
增长的数据：packed signs/byte planes、row/column scales、sparse signs/scales、indices
或 lookup。不计 activation、output、LUT shared memory、workspace 和临时 buffer。

特别注意：

- 表中的 weight bytes 是按源码 load 位置和 CTA 重复读取计算的 source-request bytes；
- 它不是 `dram__bytes_read`；cache line、L2 命中和 32-byte sector transaction 需要
  Nsight Compute 才能确定；
- 同一 warp 广播同一地址时按一次源 load 指令和对应对象字节计算，不乘 32 lanes；
- control/address 无法从 CUDA 源码精确映射到 SASS，单独作为 estimate。

## 3. 当前 artifact 的真实字节数

`deployment.py` 生成的完整 layout 同时保留 `indices` 和 `lookup`。固定 shape 下：

| artifact field | dtype/shape | bytes |
|---|---|---:|
| global codes | uint8 `[32,4096,32]` | 4,194,304 |
| global row scales | FP32 `[32,4096,2]` | 1,048,576 |
| global column scales | FP32 `[32,128,2]` | 32,768 |
| sparse codes | uint8 `[32,4096,2]` | 262,144 |
| sparse row scales | FP32 `[32,4096,2]` | 1,048,576 |
| sparse column scales | FP32 `[32,8,2]` | 2,048 |
| refinement indices | int16 `[32,8]` | 512 |
| lookup | int16 `[32,128]` | 8,192 |
| **完整 resident layout** | | **6,597,120** |

完整 layout 为约 `3.1458 bit/weight`，不是理论 2 bit/weight。dense BF16 权重为：

```text
B_dense = O * K * 2 = 33,554,432 bytes
```

不同 kernel 不会读取 layout 中所有字段：v3 使用 lookup；v4/v4_late/v5 使用 indices。
因此 resident bytes 与动态 source-request bytes 必须分开。

## 4. 动态 weight bytes

所有版本共同的 row-dependent 数据是：

```text
B_row = G * O * (
           32  global-code bytes
         + 8   global-row-scale bytes
         + 2   sparse-code bytes
         + 8   sparse-row-scale bytes)
      = 6,553,600 bytes
```

### 4.1 v3

v3 每个 64-row tile、每个 group 重新 staging column scales、sparse columns 和
lookup：

```text
bytes per (64-row tile, group)
  = 2048 global codes
  +  512 global row scales
  +  512 sparse row scales
  + 1024 global column scales
  +  128 sparse codes
  +   64 sparse column scales
  +  256 lookup
  = 4544 bytes

B_v3 = (O / 64) * G * 4544
     = 9,306,112 bytes
```

### 4.2 v4 / v4_late

`prepare_x` 将 activation 与 column scales 每 group 处理一次；主 kernel 读取所有
row-dependent artifacts：

```text
B_prepare_weight = G * (1024 global columns + 64 sparse columns + 16 indices)
                 = 35,328 bytes

B_v4 = B_row + B_prepare_weight
     = 6,588,928 bytes
```

v4_late 的 artifact 读取位置不变，因此字节数相同。

### 4.3 v5 family

v5 在每个 `(row tile, group)` CTA 内重建 34 张表。两个 sparse table bank 各自读取
一次 indices，因此每 CTA/group 的权重侧公共数据为：

```text
1024 global column-scale bytes
+ 64 sparse column-scale bytes
+ 32 index bytes
= 1120 bytes

B_v5(R) = B_row + (O / R) * G * 1120
```

| row tile R | CTA/group instances | source-request weight bytes | effective bits/weight |
|---:|---:|---:|---:|
| 256 | 512 | 7,127,040 | 3.3984 |
| 512 | 256 | 6,840,320 | 3.2617 |
| 1024 | 128 | 6,696,960 | 3.1934 |

原 v5 固定 `R=1024`，所以它和 p1024 的 weight-byte 分母相同；二者区别是 sign
layout 和运行时 bit extraction。

## 5. 循环展开与动态执行次数

### 5.1 v3

```text
row tiles             = O / 64 = 64
K splits              = G / 4  = 8
logical tasks         = 64 * 8 = 512
warps/task            = 4
rows/warp             = 16
groups/task           = 4
K16 fragments/group   = 8
```

物理 grid 最多 `108 SM * 4 = 432` CTAs，但 persistent loop 最终仍执行 512 个逻辑
tasks；动态计数必须使用 512，而不是 432。

MMA 次数：

```text
N_MMA = 512 tasks * 4 warps * 4 groups * 8 fragments
      = 65,536 warp MMA instructions
```

每 warp/group 有 64 个 `decode_weight` 源码调用位置。sparse 分支是否执行取决于真实
salient indices。没有本地 payload 可以从源码唯一确定其分布，因此用：

```text
H = active sparse decode sites per warp/group
4 <= H <= 16
uniform-index diagnostic expectation: H ~= 14.72
```

`H ~= 14.72` 仅是分布假设，不能冒充真实 artifact 的精确数字。v3 还会发出四个
有效 group staging 和两个流水线 zero-fill staging；zero-fill cp.async 是指令，但不计
有效 weight bytes。

### 5.2 v4

```text
CTA x tiles       = O / 16 = 256
K splits          = 8
CTAs              = 2048
warps/CTA         = 4
rows/warp         = 4
groups/CTA        = 4
```

对每个 `(group,row)`，v4 明确调用四次 `warp_sum`：

```text
d0, d1: two global bases
s0, s1: two sparse bases
```

因此：

```text
warp_sum calls = O * G * 4
               = 4096 * 32 * 4
               = 524,288
```

源码将 reduction 完全展开为 offsets `16,8,4,2,1`。每次 `warp_sum` 源码级恰好：

```text
5 SHFL + 5 FP32 ADD
```

所以：

```text
SHFL_v4          = 524,288 * 5 = 2,621,440
reduction FADD   = 524,288 * 5 = 2,621,440
```

由于本机没有 nvcc/cuobjdump/nvdisasm 和 cubin，无法把“编译后恰好 5+5”标为
SASS-confirmed。`__shfl_down_sync`、显式 `__fadd_rn` 和 unroll 强烈约束编译器，源码
预期是 5 SHFL + 5 FADD，但最终仍需 SASS 验证。

### 5.3 v4_late

v4_late 先在每个 lane 内应用四个 row-scale FMA，并跨当前 split 的所有 groups 累积，
最后才对每个 `(split,row)` 做一次 `warp_sum`：

```text
warp_sum calls = O * splits
               = 4096 * 8
               = 32,768

SHFL_v4_late   = 32,768 * 5
               = 163,840
```

相对 v4：

```text
2,621,440 / 163,840 = 16x fewer SHFL
SHFL reduction      = 93.75%
```

下降倍数正好等于每 split 的 `4 groups * 4 bases = 16`。

### 5.4 v5 / v5_p256 / v5_p512 / v5_p1024

每 CTA 使用 256 threads，即 8 warps，并构造：

```text
32 global LUTs = 2 bases * 16 eight-weight chunks
 2 sparse LUTs = 2 sparse bases
34 tables total
```

单张表的 `build_lane` 源码工作量为：

```text
8  seed FP adds
3  increment FP adds
7  recurrence FP adds
7  shared loads
8  shared stores
8  FP multiplies outside build_lane
```

不同 row tile 的 CTA/group 和 table build 次数：

| row tile | CTA/group instances | table builds |
|---:|---:|---:|
| 256 | 512 | 17,408 |
| 512 | 256 | 8,704 |
| 1024 | 128 | 4,352 |

但是 row compute 的动态 warp iterations 不随 row tile 变化：

```text
N_warp_row = (O/R) * G * 8 warps * (R/256 rows per thread)
           = O * G / 32
           = 4096
```

每次 warp-row iteration 读取 32 个 global LUT entries、2 个 sparse LUT entries，
进行 32 次 FP add 和 4 次 FMA。

### 原 v5 的 pattern/de-interleave

`pattern()` 的字面源码为：

```text
4 AND + 3 shift + 3 OR = 10 integer source operations/call
```

每个逻辑 `(group,row)` 调用 34 次：32 次 global、2 次 sparse。按 warp-inst 计数，
32 个不同 row lanes 同时执行相同 instruction site，因此使用 4096 个 warp-row
iterations，而不是 `O*G`。

原 v5 每个 warp-row iteration 的显式 extraction work：

```text
global pattern work = 32 * 10 + 16 pre-shifts + 8 halfword shifts = 344
sparse pattern work =  2 * 10 +  1 shift                     = 21
total               = 365 integer sites
```

planar byte-plane 版本为：

```text
global masks/shifts = 32 masks + 24 non-zero shifts = 56
sparse mask/shift   = 2
total               = 58 integer sites
```

所以源码字面删除：

```text
(365 - 58) * 4096 = 1,257,472 dynamic warp integer sites
```

编译器可能用 LOP3、SHF 或 BFE 融合这些表达式。没有 SASS 时，合理但非精确的范围是：

```text
pattern() compiled estimate       = 7--10 instructions/call
planar byte extraction estimate   = 1--2 instructions/extraction
deleted dynamic integer estimate  = 937,984--1,257,472
```

## 6. 指令分类与总表

“core”包含源码可见 global/shared memory、integer/bitwise、FP、SHFL、MMA，以及
明确的 conversion/select。control/address 使用：

```text
A_k ~= 1--2 * dynamic global/shared memory instruction sites
```

这是启发式 estimate，不是编译器或 profiler 数字。下表 `warp-inst` 使用
`core + A_k`。单位 `M` 表示一百万条动态 warp instructions。

| kernel | weight bytes | warp-inst | warp-inst/weight | warp-inst/weight-byte | SHFL inst | INT/bit inst | FP inst | MMA inst | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense BF16 | 33,554,432 | actual unknown; ideal core floor 0.635--1.094M | floor 0.038--0.065 | floor 0.019--0.033 | unknown | unknown | unknown | unknown | PyTorch/cuBLAS implementation and SASS unavailable；floor 不能当成实测 |
| v3 | 9,306,112 | **11.589--15.826M est.** | 0.691--0.943 | **1.245--1.701** | 0 | 4.162--4.850M | 1.705--2.098M | 0.0655M | sparse active sites `H=4--16`；三阶段 pipeline |
| v4 | 6,588,928 | **14.653--16.277M est.** | 0.873--0.970 | **2.224--2.470** | 2.621M | 4.588M est. | 4.196M | 0 | 每 `(group,row)` 四次 warp_sum |
| v4_late | 6,588,928 | **9.738--11.362M est.** | 0.580--0.677 | **1.478--1.724** | 0.164M | 4.588M est. | 1.738M | 0 | 每 `(split,row)` 一次 warp_sum |
| v5 | 6,696,960 | **2.103--2.834M est.** | 0.125--0.169 | **0.314--0.423** | 0 | 1.143--1.560M est. | 0.265M | 0 | R=1024；运行时 pattern/de-interleave |
| v5_p256 | 7,127,040 | **2.731--3.554M est.** | 0.163--0.212 | **0.383--0.499** | 0 | 0.400--0.499M est. | 0.604M | 0 | CTA 最多，LUT 重建最多 |
| v5_p512 | 6,840,320 | **1.687--2.236M est.** | 0.101--0.133 | **0.247--0.327** | 0 | 0.270--0.368M est. | 0.378M | 0 | q/k/v/o 实测最优；occupancy/LUT 折中 |
| v5_p1024 | 6,696,960 | **1.165--1.577M est.** | 0.069--0.094 | **0.174--0.235** | 0 | 0.205--0.303M est. | 0.265M | 0 | 最低 `I_w`；大 O/K shape 实测更优 |

更细的内存指令分类如下。所有数字仍是 source-level dynamic estimate：

| kernel | global LD | global ST | shared LD | shared ST | known core total |
|---|---:|---:|---:|---:|---:|
| v3 | 0.048M | 0.005M | 2.507--2.703M | 0.006M | 9.023--10.301M |
| v4 | 0.993M | 0.033M | 0.524M | 0.074M | 13.029M |
| v4_late | 0.993M | 0.033M | 0.524M | 0.074M | 8.114M |
| v5 | 0.104M | 0.004M | 0.170M | 0.035M | 1.790--2.208M |
| v5_p256 | 0.319M | 0.004M | 0.261M | 0.139M | 2.007--2.106M |
| v5_p512 | 0.176M | 0.004M | 0.200M | 0.070M | 1.237--1.336M |
| v5_p1024 | 0.104M | 0.004M | 0.170M | 0.035M | 0.852--0.951M |

v5 family 的 known-core total 还包含未在主表单列的 conversion/select：v5 和 p1024
约 0.069M，p512 约 0.139M，p256 约 0.280M。

## 7. 版本间下降比例

### v4 -> v4_late

```text
SHFL:       2,621,440 -> 163,840
reduction:  16x
drop:       93.75%

known core: 13.029M -> 8.114M
drop:       37.73%

total I_w including address/control estimate:
drop:       approximately 30.2%--33.5%
```

### v5 -> best byte-plane

p1024 是最低 `I_w` byte-plane 版本。它相对原 v5 删除全部运行时 pattern
de-interleave；row tile、LUT 数量和 weight-byte 分母相同。

```text
literal source integer sites removed: 1,257,472
compiled-site estimate removed:        937,984--1,257,472
known-core reduction estimate:          52.4%--57.0%
total I_w reduction estimate:           38.8%--49.9%
```

这些下降比例受 compiler fusion 和 address/control 模型影响，不能标为精确 SASS 结果。

## 8. A100 rough issue roofline

按题设和服务器记录的 A100 SXM4 最大时钟 1.410 GHz：

```text
generic issue roof
  = 108 SM * 4 schedulers/SM * 1.410 GHz
  = 609.12 G warp-inst/s

HBM roof
  = 2039 GB/s

I_roof
  = 609.12 / 2039
  = 0.2987 warp-inst/byte
```

诊断解释：

- v3、v4、v4_late 远高于 0.299，明显有 instruction-issue 压力；
- 原 v5 `0.314--0.423`，大致位于边界上方；
- p256 `0.383--0.499`，LUT 重建使其仍偏 issue-heavy；
- p512 `0.247--0.327`，跨越 rough 分界；
- p1024 `0.174--0.235`，是唯一稳定位于分界下方的 FluxBin 版本，有机会重新进入
  HBM/latency/occupancy 主导区间；
- dense 的实际动态指令未知，不能用 ideal floor 判断实际 cuBLAS bottleneck。

例如 p1024 的源码请求字节对应理想 HBM 时间约：

```text
6,696,960 / 2039 GB/s ~= 3.28 us
```

实测约 15 us，说明 roofline 只提供下界；它不能识别 shared-memory conflict、低 CTA
并行度、依赖链和实际 memory transactions。

## 9. SASS 验证状态

本机为 macOS/Apple Silicon，没有 `nvcc`、`cuobjdump`、`nvdisasm` 或 NVIDIA GPU，
因此没有生成 SASS-grounded dynamic estimate。现有服务器日志只有 ptxas resource
summary：

- v4_late：48 registers，1088 B shared，无 spill；
- v5：48 registers；
- p256/p512：40 registers；
- p1024：44 registers；
- 均未保存可供本机反汇编的 cubin。

因此本文严格区分：

```text
A. source-level dynamic estimate: 已完成
B. SASS-grounded dynamic estimate: unavailable
```

后续若需要闭合 B，应保存每个 specialization 的 cubin，并用 nvdisasm 统计实际展开、
predication、LOP3/SHF/BFE 融合、load vectorization 和 FMA fusion；最终动态次数仍需结合
循环 trip count 或 profiler counter，而不能只数静态 SASS 行数。

## 10. 与 FluxBin 论文数据的关系

本地论文为 sibling workspace `literature/fluxbin.pdf`。论文说明实验使用单张 A100
80GB，但没有明确 A100 PCIe/SXM SKU、时钟、功耗模式和完整 operator timing protocol。

4096x4096 数据：

| reference | configuration | latency |
|---|---|---:|
| paper Table 10 | pure `2b-g128`, best `Mtile=128,Ktile=32` | 41.2 us |
| paper Table 10 | pure `2b-g128`, `Mtile=128,Ktile=128` | 47.5 us |
| paper Table 11 | hybrid `2b-s8-g128`, best/default tile | 47.1 us |
| paper Table 12 | hybrid `2b-s8-g128`, rounded | 47 us |
| this project v3/gps4 Graph | A100 SXM4 | 47.247 us |
| this project v4/gps4 Graph | A100 SXM4 | 48.128 us |
| this project v4_late/gps4 Graph | A100 PCIe | 26.192 us |
| this project v5/gps1 Graph | A100 PCIe | 17.666 us |
| this project v5_p512/gps1 Graph | A100 PCIe | 14.886 us |
| this project v5_p1024/gps1 Graph | A100 PCIe | 15.389 us |

正确语义对照是 Table 11 的 hybrid `2b-s8-g128`，而不是只看 Table 10 的 pure 2b：

```text
v3 versus paper hybrid:  +0.31% latency
v4 versus paper hybrid:  +2.18% latency
```

所以“旧 v3/v4 已达到论文 FluxBin hybrid kernel 的同一延迟量级”成立。新 v5
byte-plane 数字明显更低，但来自另一台 A100 PCIe 和不同 harness，不能正式声明相对
论文 3x 加速。

论文 artifact 也不同。Section C.2 假定 scales 和 indices 为 FP16；按其公式，
4096x4096 hybrid-s8 artifact 为：

```text
paper artifact = 5,522,944 bytes = 2.6335 bit/weight
current layout = 6,597,120 bytes = 3.1458 bit/weight
```

当前 layout 多 19.45%，主要因为 row/column scales 是 FP32，并同时保留 indices 和
lookup。因此相近 latency 不证明两者 `I_w` 相同；论文未公开动态 warp-inst 或 SASS。

### 论文 5.92x 声称的口径问题

论文的 5.92x 来自 Table 1 的端到端吞吐相除，不是 Table 10：

```text
254.39 tokens/s / 42.94 tokens/s = 5.92x
```

但 Table 12 与该吞吐没有明显自洽。其 hybrid latency 为 4096x4096 的 47 us 和
4096x11008 的 109 us。LLaMA-2-7B 有 32 层，即便每层只计 o_proj 和 down_proj：

```text
(47 + 109) us/layer * 32 layers = 4.992 ms/token
upper bound                           = 200.3 tokens/s
```

这尚未计 Q/K/V、gate/up、attention、norm、KV cache 和 sampling，却已低于论文
254.39 tokens/s 对应的 3.93 ms/token。论文可能存在 tokens/s 分子、batch、融合、
CUDA Graph、isolated timing 或 FP16 baseline 的未公开口径差异；仅凭 PDF 无法闭合。
因此 5.92x 应表述为“论文报告的端到端比值”，不能视为由 Table 10/12 独立验证的
kernel speedup。

## 11. W4A16 后期竞争基线

`NewSmallProject` 的正式 W4A16 结果不是 A100/Qwen3-8B/M=1 Linear：

| item | current FluxBin | W4A16 evidence |
|---|---|---|
| GPU | A100 SXM4 80GB, SM80 | Isambard GH200 120GB, Hopper SM90 |
| model | Qwen3-8B, 36 layers | LLaMA-2-13B, 40 layers |
| shapes | 4096/12288, GQA K/V=1024 | 5120/13824 |
| backend | prepared HF + CUDA Graph | vLLM 0.25.1 + MacheteLinearKernel |
| workload | batch 1, fixed 32-step decode | 256 input + 64 output, c1/c8 |
| main speedup | 1.381--1.383x | c1 1.53--1.54x; c8 about 1.37x |

W4A16 的正式 serving 设置 `enforce_eager=false`，允许 vLLM 使用优化/Graph 路径。
只有 layer-0 diagnostic 使用 `enforce_eager=true`，其 decode 仅快约 1.01--1.04x，
不能替代完整 serving 结果。

W4A16 可以作为实验后期必须超过的工程基线，但当前不能横向判胜。若暂时把 c1
`1.533x` 当作诊断目标，FluxBin 当前约 `1.382x`：

```text
speedup gap                         ~= 10.9%
same-baseline packed latency needed ~= 9.8% further reduction
```

若按 W4A16 c1 TPOT `8.684 -> 5.510 ms`，即 `1.576x`，则 packed latency 还需约
12.3% 的进一步下降。由于硬件、模型和后端不同，这些不是正式验收门槛。

后期公平迁移应分两级：

1. 在同一 A100 上，为相同 Qwen3-8B revision 构建 W4A16，对相同七种真实 shape、
   M=1、dtype、输入和计时协议进行算子比较，并记录实际选中的 backend kernel；
2. FluxBin 与 W4A16 进入同一个 serving/sequence harness 后，再比较相同 prompt、KV、
   concurrency、output length、同轮 BF16 baseline 的 TPOT/E2E/throughput。

迁移属于实验后期，本轮不执行。

## 12. 当前优化方向

### 已能确定

1. 保留 `v5 byte-plane LUT` 主线，不回到 v4 的 per-group warp reduction，也不再把
   pattern/de-interleave 当成主要优化空间。
2. 使用 shape-aware row tile：当前数据支持 K=4096 且 O<=4096 的 q/k/v/o 使用
   p512；O=12288 或 K=12288 的 MLP 使用 p1024。
3. 该 dispatch 将七个 Linear Graph 合计从约 124.598 us 降至约 120.651 us，
   Linear 级约 3.17%，是当前最确定、风险最低的收益。
4. p512 在 q_proj 上比静态 `I_w` 更低的 p1024 快，说明下一阶段应优先检查
   CTA/SM 数量、occupancy、tail、LUT build、shared bank conflict 和 split reduction，
   而不是继续只减少整数指令。

### 仍需 profiler 才能确定

- shared-memory LUT bank conflicts 的实际占比；
- long scoreboard、dependency chain 和 issue utilization；
- L2/HBM 实际 bytes 与 cache sector amplification；
- split workspace 写入和 finish reduction 的实际成本；
- p512/p1024 的 active CTA、tail 和 occupancy。

### 独立的高潜力 artifact 分支

将 FP32 scales 改为 FP16 storage 可把 resident artifact 从约 6.60 MB 降至约
5.55 MB，节省约 15.9%，接近论文 2.63 bit/weight。但这会改变 artifact 和数值合同，
必须重新进行 Linear、完整模型、PPL 和部署验证，不能混入当前纯调度优化。

### 为什么还不能只靠 dispatch 超过 W4A16

当前完整模型平均约：

```text
BF16   ~= 14.61 ms/token
packed ~= 10.58 ms/token
speedup ~= 1.382x
```

七个 p1024 Linear 粗略贡献：

```text
124.598 us/layer * 36 layers ~= 4.49 ms/token
```

shape-aware dispatch 只节省约 `0.14 ms/token`。若其他部分完全不变，要达到诊断性的
W4A16 c1 `1.533x`，Linear 总时间还需约 23% 的下降；达到 `1.576x` TPOT 目标则约需
29%。因此最终需要 inner kernel、split/reduction、wrapper/launch 和完整执行路径共同
优化，不能只依赖 row-tile dispatch。

## 13. 最终判断

按 `I_w` 排名：

```text
v5_p1024 < v5_p512 < v5 ~= v5_p256 << v3 ~= v4_late < v4
```

v5 与 p256 的顺序可能被 compiler/address 细节改变；p1024 最低则不依赖该细节。
性能选择不能直接照抄此排名：q_proj 的实测最优是 p512，而大 O/K shapes 更适合
p1024。

当前可冻结的工程方向是：

```text
v5 byte planes
+ shape-aware p512/p1024 dispatch
+ A100 profiler 决定 shared-LUT / occupancy / split-reduction 的下一步
+ 实验后期迁移同机 W4A16 竞争基线
```

## 14. 证据入口

- CUDA source：`src/fluxbin_style/csrc/m1_v3.cu`、`m1_v4.cu`、
  `m1_v4_late.cu`、`m1_v5.cu`、`lut8.cuh`
- artifact/layout：`src/fluxbin_style/deployment.py`
- 当前 Linear/full-model 数据：`docs/QWEN3_8B_M1_LINEAR_RESULTS.md`
- PCIe candidate 数据：
  `server_results/runpod_m1_inner_pcie_2026-09-15/local-summary/summary.md`
- ptxas resource log：`server_results/runpod_m1_inner_pcie_2026-09-15/tests.log`
- FluxBin paper：sibling workspace `literature/fluxbin.pdf`
- W4A16 结果：sibling workspace `NewSmallProject/docs/VLLM_W4A16_RESULTS.md`
