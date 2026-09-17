# 当前进度与交接

更新：2026-09-17。M=1 decode 的现有 QBB 加速结果已经冻结。Qwen3-8B uniform
symmetric GPTQ W3 g128 与当前 QBB 的同协议质量/码率四臂对照已完成并验收，
结论为 **Case A：后续优先 uniform 3-bit backend / solver**。A8 不在本轮范围，
vLLM 仅保留接口、尚未接入。

## 当前结论

- **GPTQ W3 inline LUT 第一轮 4×3 Linear trial 已完成**：A100 80GB PCIe，
  同轮 CUDA Graph total，12/12 cell 正确且稳定。最佳 q/o R256、k/v R512、
  gate/up R1024、down R1024 相对原始 BF16 分别为 1.731x、0.982x、2.678x、
  2.616x；详见 [W3 inline 结果](W3_LUT_INLINE_RESULTS.md)。
- **W3 完整模型性能约 1.49x，但 correctness gate 未通过**：A100 PCIe、batch1、
  真实前缀 StaticCache、32-token full-sequence Graph 相对原始 BF16 为
  1.4880x / 1.4892x，相对 decoded W3 为 1.4923x / 1.4886x；主计时稳定。
  packed-vs-decoded logits NRMSE 为 0.01683 / 0.01457，超过 0.005，因此状态是
  `completed_with_backend_numerical_differences`，不是 accepted full-model backend。
- 逐层诊断证明 prefill、36 层 prefix KV、prepared wrapper 和 Graph 路由无误；
  第 0 层已有约 0.0015 NRMSE，第 8 层首次越过 0.005。四种真实 shape 均与
  structural reference 逐位一致；问题已缩小到 kernel 的“FP32 integer dot 后乘
  scale”与 decoded 路径“先物化 BF16 权重再 GEMM”之间的有限精度差异，但现有
  四-shape 对照尚未通过逐舍入点因果消融确定主导项。详见
  [W3 完整模型结果与误差归因](W3_LUT_FULL_MODEL_RESULTS.md)。
- 当前服务器的 Nsight Compute 计数器权限被宿主拒绝（`ERR_NVGPUCTRPERM`）；
  尚不能归因 LUT build、occupancy、HBM 或 shared bank conflict。prepare 诊断分支
  仍未授权、未实现。
- **两条 decoded-BF16 corrected 完整模型路线已在 A100 PCIe 跑完**：按 10 次
  CUDA event 中位数，`fast_corrected` 相对原始 BF16 为 1.3441x / 1.3427x，
  `observed_exact` 为 1.2703x / 1.2699x。两者都低于历史 structural 路线的约
  1.49x；性能优先时继续以 structural 为主线。协议的全比较稳定门因 isolated
  outlier 未通过，所以这些数值是 raw median 观测，不是 formal-stable acceptance；
  `fast_corrected` 自身两组 CUDA device 计时均稳定，但 wall-time 有 host-side outlier。
- 两条 corrected 路线的 correctness 仍失败：`fast_corrected` logits NRMSE 为
  0.02076 / 0.02200，`observed_exact` 为 0.01973 / 0.01702，均超过 0.005。
  prepared wrapper、Graph 和 252 条 packed route 全部精确且无 dense fallback，
  因而失败不是路由问题。`±3` 修正复现显式 FP32 grouped matvec 的 decoded-BF16
  物化语义，但不能复现 cuBLAS BF16 GEMM 的归约树；36 层传播后仍会放大差异。
  详见 [A100 corrected full-model 记录](W3_CORRECTED_FULL_MODEL_RUNBOOK.md)。

- **加速的主要障碍已不在 Linear（重读已有 JSON，非新运行）**：252 个 Linear 占
  batch-1 decode 的 62.1%（9.263 ms/token），lm_head 5.8%，其余 32.1%
  （4.787 ms/token）是小算子 kernel 数量乘固定延迟。代入每层 Linear 2.205x 预测整模
  1.514x，实测 1.488x，误差 1.8%。硬上限：Linear 归零 2.64x，Linear 达 BF16 同等
  带宽约 1.98x。**不动这 32% 就到不了 2x 以上。** 详见
  [非 Linear 路径融合](NONLINEAR_FUSION.md)。
- **0.005 数值门在 harness 噪声地板以下（重读已有 JSON，非新运行）**：零量化的
  `original_bf16` 臂在数学等价的 mask 改写下 logits NRMSE 已是 0.01431 / 0.01153，
  同样不通过；QBB 线独立复现同一地板 0.01167--0.01359。放大链条为逐 Linear
  0.00059--0.00264 →（深度约 4x）step-0 logits 0.0085596 →（自回归约 2x）step-32
  0.01683。**上述几处 correctness 失败因此不能读作 kernel 算错**，也解释了 `±3`
  修正为何 Linear 层面有效而端到端更差。详见
  [数值门标定](W3_NUMERICAL_GATE_CALIBRATION.md)。
- **已实现、尚未在 GPU 验证**：(a) 自校准相对数值门
  `packed_vs_decoded_relative_check`（以同轮 `original_bf16` 的 dyn-vs-static 为地板，
  limit 由 protocol 声明、默认 1.0）与逐步 NRMSE `stepwise_normalized_rmse`；
  (b) 鲁棒化计时门 `trimmed_relative_range`（去首尾各一样本，median 仍用全样本，
  故加速比数值不变）；(c) Qwen3 RMSNorm/RoPE 融合；(d) shared-memory bank conflict
  计时探针。166 项测试通过。
- 用新门重放归档结果：09-16 structural 的判定与加速比**完全不变**；09-17 的三个
  cell 从 void 恢复为可发布的负面性能结果，`fast_corrected` p1 因 wall-time 有两个
  尖峰仍判不稳。相对门下所有 cell 仍不通过，ratio 1.18--1.91——这是有意义的负面
  结果，不是相对不可达阈值的 3.4 倍。
- 需要注意的反例：`torch.nn.functional.rms_norm` **不是**融合 kernel。
  `aten::rms_norm` 是 CompositeImplicitAutograd、无后端注册，CUDA 上分解成同样 8 个
  op；torch 2.9 才有 `_fused_rms_norm`，服务器是 2.8。因此 RMSNorm 融合走 Inductor，
  该事实已固定为回归测试。

- **完整模型 sequence Graph 已实现 1.381x / 1.383x 原始 BF16 加速**：
  A100 SXM4 80GB，`v5_p1024/gps1`，prepared 协议 v2.1，两个固定 prompt。
  32 token 耗时约 468 → 338 ms，约 68 → 95 token/s。
- Graph 两组均稳定，wall/device 相对极差 <0.12%。同轮 eager 波动 6.51–10.11%，
  超过 5% 门槛，不发布 eager 加速比；整体状态因此为 `completed_unstable`。
- 113/113 GPU 测试通过，runner 修正后专项 7/7 复验通过；六组 arm/prompt 的
  checked/prepared wrapper、Graph/eager 输出精确一致，全部正式重复一致。
- packed/decoded 与动态/静态注意力数值差异均按 report-only 保留。
  这些性能结果不代表 BF16 数值等价、自由生成质量或 vLLM 服务吞吐。
- step400 test PPL 为 13.169788495，BF16 为 9.724944981；原质量门槛未通过，
  用户已授权加速研究先行。当前 PPL 证据来自解码后的 dense BF16 路径。
- RTX PRO 4500 同卡四臂 PPL：BF16 9.726488、当前 QBB 13.167910、GPTQ W3
  11.266115、QBB FP16-scales 13.168951；全部覆盖 36 layers / 252 Linears /
  6,945,767,424 weights、146 blocks 和 298,862 positions。
- GPTQ 实际 3.154552 bit/weight，只比当前 QBB 3.138184 高 0.5216%，PPL 却低
  1.9018（14.4426%）。QBB FP16 scales 为 2.631687 bit/weight，PPL 几乎不变。
  按冻结规则属于 Case A：停止继续为当前 QBB format 做深度 kernel 优化。

近期各版本对照与证据：[结果总览](RESULTS_OVERVIEW.md)、
[Linear / block / 全模型详细结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。

## 有效代码与协议

| 项目 | 当前入口 |
|---|---|
| packed kernel | `src/fluxbin_style/csrc/m1_v5.cu`，`v5_p1024/gps1` |
| prepared Linear 包装 | `src/fluxbin_style/deployment.py` |
| 静态 KV / 整段 Graph | `src/fluxbin_style/static_decode.py` |
| 当前全模型 runner | `scripts/run_qwen3_8b_prepared_m1_trial.py` |
| 当前配置 | `configs/acceleration/qwen3_8b_full_m1_v2.json`，内部 ID 为 v2.1 |
| 旧协议对照 | `scripts/run_qwen3_8b_full_m1_trial.py` + `qwen3_8b_full_m1_v1.json` |
| W3 packed kernel | `src/fluxbin_style/csrc/w3_lut.cu` |
| W3 prepared Linear | `src/fluxbin_style/w3_lut_deployment.py` |
| W3 全模型 runner | `scripts/run_qwen3_8b_w3_full_m1_trial.py` |
| W3 全模型配置 | `configs/acceleration/qwen3_8b_w3_full_m1_v1.json` |
| W3 corrected 全模型配置 | `configs/acceleration/qwen3_8b_w3_corrected_full_m1_v1.json` |
| 非 Linear 融合 | `src/fluxbin_style/fused_modules.py` |
| W3 fused 全模型配置 | `configs/acceleration/qwen3_8b_w3_fused_full_m1_v1.json` |
| bank conflict 探针 | `scripts/run_w3_lut_bank_conflict_probe.py`（计时专用，输出数值无效） |
| GPU 验证批次 | `scripts/run_w3_gpu_validation_batch.sh` + `scripts/summarize_w3_gpu_validation_batch.py` |

QBB prepared v2.1 最新正式运行源码为 `3c996ab`。v2 首次在动态/静态数值门槛处失败，未计时；
v2.1 将注意力路径差异独立报告，以同一 StaticCache 下的旧 wrapper 为精确参照。
失败记录与修正后的结果均保留，不追溯改写 v1 或首次 v2 结果。

W3 完整模型正式运行源码为 `fc76b75`。四个最佳 row tile、layout manifest、
protocol、runner、environment 和正式结果均由 SHA256 绑定；诊断只读正式 artifact，
没有改写正式 JSON 或阈值。

W3 corrected 完整模型运行源码为 `d0f85b4`。protocol、runner、layout manifest、
environment 和结果分别绑定到 SHA256 `f7d3f12c...56b24`、
`511a9ac7...054e`、`f2825dda...edf8`、`9f045ce5...091e` 和
`08271b47...c0cd`；正式状态为
`completed_with_backend_numerical_differences`。

## 服务器与证据

W3 完整模型服务器最后一次观测：`213.173.105.10:43680`，A100 80GB PCIe。
正式任务与两项误差诊断均退出 0；最后检查无 GPU compute 进程或 tmux，服务器 Git
工作区干净且为 `fc76b75`。正式结果 SHA256 为
`c68811aa...56f423`，逐层/算术归因 SHA256 分别为 `86208468...7cbe6` 和
`f1767d0e...4a85`。远端结果位于 `/workspace/results/qwen3-8b-w3-full-m1-v1/`，
私有本地备份位于 `server_results/runpod_w3_full_m1_a100_pcie_2026-09-16/`。
可以关闭计算实例并保留 `/workspace` 网络卷；实际电源状态仍由用户确认。

W3 corrected 服务器最后一次观测：`213.173.105.8:42353`，A100 80GB PCIe。
retry1 于 2026-09-17 02:25:55Z 启动、03:00:14Z 完成，退出 0；最后检查 GPU、
tmux 和实验进程均为空。远端结果位于
`/workspace/results/qwen3-8b-w3-corrected-full-m1-v1-a100-pcie-20260917-retry1/`，
本地私有备份位于
`server_results/runpod_w3_corrected_full_m1_a100_pcie_2026-09-17/`，结果 SHA256
为 `08271b47c7db3e5197557a6fef25af659cf90e885621e7d4660a99d3c3a2c0cd`。
首个启动因 tmux quoting 选到 system Python，在模型加载前失败；失败日志保留，retry1
改用绝对 venv Python 后完成。实例已完成关机准备并保留 `/workspace`；实际电源状态
仍由用户确认。

M=1 性能服务器最后一次观测：`213.173.102.5:11028`，A100 SXM4 80GB，任务退出 0，
GPU 无剩余实验进程；已完成关机准备，**尚无用户确认本实例已关闭**。
本说明更新未重新连接服务器，不推断当前电源状态。

- 网络卷：`34au39ljvf`，挂载 `/workspace`；关闭计算实例时保留卷。
- payload：`/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`。
- 模型 revision：`b968826d9c46dd6066d109eabc6255188de91218`；完整恢复仍需原 snapshot。
- 结果 / 作业：`/workspace/results/m1-sxm4-20260915-prepared/`、同名 `/workspace/jobs/` 目录。
- 私有本地备份：`server_results/runpod_prepared_sxm4_2026-09-15/`；哈希、120 次测量和路由已核验。
- 环境：torch 2.8.0+cu128、CUDA 12.8、Transformers 5.14.1；容器本地 venv，
  持久保存包/模型/兼容编译缓存。本轮依赖恢复约 40 秒。基础镜像 digest 仍未知。

详见[恢复手册](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)与[关机交接](SERVER_SHUTDOWN_READY.md)。
原始结果、权重、缓存不提交 GitHub。

质量实验服务器最后一次观测：`213.173.105.8:48380`，A100 80GB PCIe。
GPTQ 首次量化完成后因 GPTQModel 7.4.0 禁用 `split_by='layer'` 而在保存阶段
退出 1；完整 252-module disk offload 已由 `63a4bc8` 无重新量化恢复为标准
5-shard checkpoint，并成功走标准 GPTQModel reload 和 36-layer BF16 decode。
recovery 退出 0，GPU/实验进程和 tmux 均为空，可关闭计算实例并保留网络卷。
精确路径、哈希和恢复边界见
[W3/QBB artifact 状态](QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

质量实验服务器最后一次观测：`213.173.109.240:44534`，RTX PRO 4500
Blackwell 32GB、CC 12.0、torch 2.8.0+cu128。作业 revision `f979f6b`，退出 0，
summary JSON SHA256 为 `de5e12f...e88e8cd`；tmux、GPU 和实验进程均为空。
结果、作业记录及关机归档均保存在网络卷 `34au39ljvf`；私有本地备份位于
`server_results/runpod_w3_rate_distortion_rtx4500_2026-09-16/`，归档 SHA256
为 `812ba530...49bfe7`。可关闭计算实例并保留网络卷。精确哈希和路径见
[W3/QBB artifact 状态](QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

本次是 RTX 同卡质量对照，不是 A100 精确复现或性能结果；旧 A100 两个 anchor
仅 report-only。默认 `formal-a100` 路径仍保留，除非明确要求复现，否则不再重跑。

## 下一步边界

已完成的 RTX PRO 4500 诊断 46/46 cell 通过 correctness、repeat 与 Graph 检查。
`fast_corrected` 的四类 shape 均选 gps1；`observed_exact` 选 gps1/1/4/2。诊断结果
SHA256 为 `4f567a9adf28d80eb2f7d08f08146a08a855d80f3c9589f02d5bd845f51460ca`，但这是
Linear/layer-0 固定输入证据，不是 A100 或完整模型结论。RTX 实例已关闭并保留网络卷。

corrected 路线现已冻结为负面性能/正确性 follow-up，不再自动重跑。其中
`observed_exact`（gate/up gps4、down gps2）已在 A100 上比 gps1 慢 5.80% / 5.73%，
arithmetic 相同故可干净归因到 gps：gate/up 的 grid 从 (12,32)=384 个 block 掉到
(12,8)=96 个，低于 108 个 SM，省下的 12 个百分点 partial 流量补不回 occupancy 损失。
中间点 gate/up gps2（192 blocks）尚未测过。

**下一步是跑已准备好的 GPU 验证批次**，而不是继续改 kernel。按 62/38 分解，非 Linear
是比 kernel 更大的杠杆，且两者独立；在拿到真机数据前不宜再猜。批次含四个作业
（bank conflict 探针、A100 上的 46-cell split/row-tile sweep、同会话 stock 与 fused
全模型对照），fail-soft 且可续跑，约 90 分钟，见
[GPU 验证批次运行手册](W3_GPU_VALIDATION_BATCH_RUNBOOK.md)。批次还会复检 Nsight：
若该实例允许硬件计数器，profile 的价值高于批次内任何一项。

批次最关键的两个风险点：融合是否在 CUDA Graph capture 下真的生效（若 fused 与 stock
的 `original_bf16` 每 token 基本相同，说明 torch.compile 未生效），以及 CUDA 上每次
compiled RMSNorm 实际几个 kernel（CPU 为 2，外推的 -56% 依赖它等于 1）。

批次之后按结果择一：探针若显示 conflict 占比高，kernel 重写针对查表结构而非 LUT
构建；否则转查 DRAM 访存与 occupancy。更远的两项尚未做——lm_head 量化（约省
0.66 ms/token，但该层对量化最敏感，建议先试 W8），以及把整个 `Qwen3DecoderLayer`
交给 torch.compile（53 → 约 12 kernel/层，前提是把 `PYBIND11_MODULE` 绑定的 W3 op
注册为 `torch.library.custom_op`，否则每 token 在 252 个 packed Linear 上 graph break）。

3.154552-bit W3 相对 BF16 的存储优势约 5.07x，但相对理想 W4 只有约 1.27x；A100
没有原生 INT3 MMA。若 W3 的 bitplane decode、LUT、同步和 reduction 开销超过 27%，
其带宽优势就不足以超过成熟 W4A16。若恢复上机，先做可采 performance counters 的
structural profile，再决定 main/finish fusion、persistent/fused Linear 或 Tensor Core
重构；correctness 失败继续作为边界，不得改写为 accepted——但按上文标定，现行 0.005
门在 0.005 处没有分辨力，应改用相对门或 step-0 门评判，而不是放宽绝对阈值。

每次新运行仍使用独立输出目录和匹配的源码/环境记录，保留失败证据。
vLLM 接口继续保留，接入工作尚未开始；不自动恢复 profiler、A8、精度搜索或 32B 实验。

## 历史资料

[历史交接归档](archive/HANDOFF_HISTORY_THROUGH_2026-09-15.md)保存早期 8B/32B
执行过程、哈希和验收限制；质量结果见[结果总览](RESULTS_OVERVIEW.md)，
算法说明和历史 32B 表格见[项目 README](../README.md)。
