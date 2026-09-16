# 当前进度与交接

更新：2026-09-16。M=1 decode 的现有 QBB 加速结果已经冻结。Qwen3-8B uniform
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
  structural reference 逐位一致，根因是 kernel 的“FP32 integer dot 后乘 scale”
  与 decoded 路径“先物化 BF16 权重再 GEMM”的算术语义不同。详见
  [W3 完整模型结果与误差归因](W3_LUT_FULL_MODEL_RESULTS.md)。
- 当前服务器的 Nsight Compute 计数器权限被宿主拒绝（`ERR_NVGPUCTRPERM`）；
  尚不能归因 LUT build、occupancy、HBM 或 shared bank conflict。prepare 诊断分支
  仍未授权、未实现。

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

QBB prepared v2.1 最新正式运行源码为 `3c996ab`。v2 首次在动态/静态数值门槛处失败，未计时；
v2.1 将注意力路径差异独立报告，以同一 StaticCache 下的旧 wrapper 为精确参照。
失败记录与修正后的结果均保留，不追溯改写 v1 或首次 v2 结果。

W3 完整模型正式运行源码为 `fc76b75`。四个最佳 row tile、layout manifest、
protocol、runner、environment 和正式结果均由 SHA256 绑定；诊断只读正式 artifact，
没有改写正式 JSON 或阈值。

## 服务器与证据

W3 完整模型服务器最后一次观测：`213.173.105.10:43680`，A100 80GB PCIe。
正式任务与两项误差诊断均退出 0；最后检查无 GPU compute 进程或 tmux，服务器 Git
工作区干净且为 `fc76b75`。正式结果 SHA256 为
`c68811aa...56f423`，逐层/算术归因 SHA256 分别为 `86208468...7cbe6` 和
`f1767d0e...4a85`。远端结果位于 `/workspace/results/qwen3-8b-w3-full-m1-v1/`，
私有本地备份位于 `server_results/runpod_w3_full_m1_a100_pcie_2026-09-16/`。
可以关闭计算实例并保留 `/workspace` 网络卷；实际电源状态仍由用户确认。

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

下一步不是继续调 row tile 或自动实现 prepare，而是先明确 W3 的数值语义：若要求
严格复现 retained decoded-BF16 GPTQModel，kernel 必须模拟 per-weight BF16 物化/
舍入并重做正确性和性能门；若保留当前 structural W3 算术，则将它视为另一个部署
模型，单独完成 PPL/质量验证。不得放宽 0.005/0.05 门限后把本轮改写为 accepted。

只有数值路线明确后，才决定是否在允许 performance counters 的实例上 profile 四个
最佳 inline candidate。prepare 仍只能在 profiler 证明 LUT build 为主要瓶颈后作为
诊断分支；不自动扩展 batch、Tensor Core/Marlin W3、QKV fusion、split-K 或 serving。
现有 prepared v2.1 继续作为冻结的 QBB 性能基线。

下一次上机先确认选择的是 dense-BF16 fidelity 路线还是 structural-W3 质量验证路线，
再同步 Git、核对持久卷/manifest/snapshot、恢复环境并验证 CUDA；仅 profiler 路线
需要事先确认实例允许读取 performance counters。
每次新运行使用独立输出目录和匹配的源码/环境记录，保留失败证据。
vLLM 接口继续保留，接入工作尚未开始；不自动恢复 profiler、A8、精度搜索或 32B 实验。

## 历史资料

[历史交接归档](archive/HANDOFF_HISTORY_THROUGH_2026-09-15.md)保存早期 8B/32B
执行过程、哈希和验收限制；质量结果见[结果总览](RESULTS_OVERVIEW.md)，
算法说明和历史 32B 表格见[项目 README](../README.md)。
