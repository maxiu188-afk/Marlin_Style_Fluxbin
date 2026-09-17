# 非 Linear 路径融合：分析、实现与预期

更新：2026-09-17。本页记录 Qwen3-8B batch-1 decode 中**非 Linear 部分**的开销归因和
已实现的融合。所有性能数字是基于已有 A100 测量的**外推估算**，不是新的 GPU 结果；
kernel/op 数量的下降是 CPU 上的实测。

## 为什么非 Linear 是当前瓶颈的一半

用 [Linear 级 4x3 结果](W3_LUT_INLINE_RESULTS.md)的 BF16 臂逐 shape 时间反推每层
Linear 时间，再对照[完整模型结果](W3_LUT_FULL_MODEL_RESULTS.md)：

| 项 | 每 token | 占比 |
|---|---:|---:|
| 原始 BF16 总时间 | 14.909 ms | 100% |
| 252 个 Linear | 9.263 ms | 62.1% |
| lm_head（1.245 GB BF16，未量化） | 0.859 ms | 5.8% |
| 其余小算子 | **4.787 ms** | **32.1%** |

把每层 Linear 2.205x 代入，预测整模 1.514x，实测 1.488x，误差 1.7%，分解自洽。

由此得到硬上限：Linear 时间归零也只有 **2.64x**；Linear 打到 BF16 同等带宽约
**1.98x**。**不动这 32%，就到不了 2x 以上。**

那 4.787 ms 不是带宽，是 kernel 数量乘固定延迟。stock `Qwen3RMSNorm.forward`
每次调用发 8 个 elementwise/reduction kernel，每层 4 个 norm；stock RoPE 每层 10 个。
在一个 4 层小 Qwen3 上实测一个 decode step：**249 个 compute op，每层 62.2 个，
其中 RMSNorm 占 55%**。按 36 层外推约 1900 个小 kernel，除进 4.787 ms 得每个约
2.51 us —— 正是 CUDA Graph 内 trivial kernel 的典型 GPU 侧开销。数量模型闭合。

## 一个需要记录的反例：`F.rms_norm` 不融合

直觉上 `torch.nn.functional.rms_norm` 应该是融合 kernel。**它不是。**
`aten::rms_norm` 是 CompositeImplicitAutograd、没有任何后端注册，在 CUDA 上分解成
与 CPU 完全相同的 8 个 op（用 FakeTensor 走 CUDA 分支验证过）。torch 2.9 起才有
`aten::_fused_rms_norm`，而服务器是 **torch 2.8**，根本没有这个算子。

因此 RMSNorm 的融合机制必须是 Inductor（或手写 Triton），不能是 `F.rms_norm`。
`tests/test_fused_modules.py::test_functional_rms_norm_would_not_have_fused` 把这个
事实固定为回归测试，避免以后再踩。

## 已实现

`src/fluxbin_style/fused_modules.py`，由 `fused_qwen3_modules()` 上下文管理器在
**类/模块作用域**打补丁并在退出时还原。作用域是刻意的：融合路径与 stock 不是
逐位一致，只给部分 arm 打补丁会让对照失效，所以要么全开要么全关。

**RoPE（纯张量代数，无自定义 kernel）**：cos/sin 在 head 轴上广播，所以把 q 和 k
沿 head 轴拼成一个张量后只需一次 rotate-half 和一次 `addcmul`，拆回来用 `narrow`
是 view。**10 个 op 降到 5 个**，float32 下与 stock 逐位一致。

**RMSNorm（Inductor）**：`rms_norm_reference` 逐字复制 stock 的算术（eager 下与
stock **逐位一致**，有测试保证），再由 `torch.compile` 融合。只改 kernel 数量，
不改运算或其顺序。CPU 上每次调用 2 个 inductor kernel；CUDA 上这类 persistent
reduction 通常是 1 个，**需要首轮 GPU 运行确认**。

### 实测的 op 数下降（4 层小 Qwen3，单 decode step，CPU）

| 配置 | op/step | 每层 | 相对 stock |
|---|---:|---:|---:|
| stock | 249 | 62.2 | — |
| 仅 RoPE 融合 | 229 | 57.2 | -8% |
| RoPE + RMSNorm（CPU，2 kernel/call） | 127 | 31.8 | **-49%** |
| RoPE + RMSNorm（CUDA 预期，1 kernel/call） | 110 | 27.5 | **-56%** |

### 外推到 A100 完整模型（估算，未实测）

按 -56% 作用于 4.787 ms：非 Linear 降到约 2.11 ms，加 lm_head 后 F≈2.97 ms。

| | 每 token | 吞吐 | 加速比 |
|---|---:|---:|---:|
| 原始 BF16（现状） | 14.909 ms | 67.1 tok/s | — |
| packed（现状） | 10.020 ms | 99.8 tok/s | 1.488x |
| 原始 BF16（融合后） | ~12.23 ms | ~81.8 tok/s | — |
| packed（融合后） | ~7.17 ms | ~139 tok/s | **~1.71x** |

融合同时加速两个臂，所以**绝对吞吐的提升（100 → 139 tok/s）比加速比的提升更能
说明问题**；加速比上升只是因为 F 是两臂共有的加性项。

## 运行入口

新增 `configs/acceleration/qwen3_8b_w3_fused_full_m1_v1.json`，protocol id
`qwen3-8b-w3-fused-nonlinear-full-m1-v1`，除 `fused_nonlinear_modules` 全开外与
frozen structural v1 逐字段相同（有测试比对）。两个既有 frozen protocol 显式声明
`{"rms_norm": false, "rope": false}`，行为不变。runner 把实际生效的配置写入结果的
`fused_nonlinear_modules` 字段。

融合路径与 stock 不是逐位一致，所以它是**独立的 protocol id**，不是对既有 id 的
重定义；其结果不能与 frozen structural v1 的历史数字混在一张表里。

## 未做与边界

- **CUDA 上每次 compiled RMSNorm 到底几个 kernel 未验证**，上表的 -56% 依赖它等于 1。
- torch.compile 与现有手工 CUDA Graph capture 的交互未在 GPU 上验证。capture 前有
  两次 warm-up run，编译应在那时完成，但这是推断。
- SiLU-mul 只能省 1 个 kernel/层（约 0.09 ms/token），需要自定义 kernel，性价比低，
  未做。
- `repeat_kv` 是否触发（`use_gqa_in_sdpa`）未在 GPU 上确认；若触发，每 token 多搬
  约 189 MB。
- lm_head 量化未做。W3 可省约 0.66 ms/token，但 lm_head 对量化最敏感，建议先试 W8。
- 把整个 `Qwen3DecoderLayer` 交给 torch.compile 收益更大（53 → 约 12 kernel/层），
  但前提是把 `PYBIND11_MODULE` 绑定的 W3 op 注册成 `torch.library.custom_op`，
  否则每 token 会在 252 个 packed Linear 上 graph break。未做。
