# 当前进度与交接

更新：2026-09-15。M=1 decode 的现有加速结果已经冻结。Qwen3-8B uniform
symmetric GPTQ W3 g128 与当前 QBB 的同协议质量/码率对照已完成量化 artifact
准备和四臂输入预检，PPL 尚未启动。A8 不在本轮范围，vLLM 仅保留接口、尚未接入。

## 当前结论

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
- W3 GPTQ 已覆盖 36 layers / 252 Linears / 6,945,767,424 weights，实际
  `qweight + scales + qzeros + g_idx` 为 2,738,847,744 bytes，即
  3.154551630 bit/weight。QBB FP16-scale artifact 也已完成，实际 tensor bytes
  为 2,284,886,016。两者和冻结 BF16/QBB 输入已通过四臂 `--validate-only`，
  但没有 PPL，暂不能决定 uniform W3 与 binary-base 路线。

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

最新正式运行源码为 `3c996ab`。v2 首次在动态/静态数值门槛处失败，未计时；
v2.1 将注意力路径差异独立报告，以同一 StaticCache 下的旧 wrapper 为精确参照。
失败记录与修正后的结果均保留，不追溯改写 v1 或首次 v2 结果。

## 服务器与证据

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

## 下一步边界

下次上机直接执行 [W3/QBB 同码率质量对照](QWEN3_8B_W3_RATE_DISTORTION_RUNBOOK.md)
的 Stage 3，不重新量化或重建 QBB FP16-scale artifact；用四臂 PPL 决定继续
binary-base 还是转向 uniform 3-bit。在该结果出来前不继续扩展 kernel 候选。
已有 prepared v2.1 仍是冻结性能基线。单 block 速度不作为完整模型启动前提，
跨 GPU、KV、驻留与计时协议的结果不得直接作单因素因果比较。

下一次上机先同步 Git、核对持久卷/manifest/snapshot，再恢复环境、验证 CUDA。
每次新运行使用独立输出目录和匹配的源码/环境记录，保留失败证据。
vLLM 接口继续保留，接入工作尚未开始；不自动恢复 profiler、A8、精度搜索或 32B 实验。

## 历史资料

[历史交接归档](archive/HANDOFF_HISTORY_THROUGH_2026-09-15.md)保存早期 8B/32B
执行过程、哈希和验收限制；质量结果见[结果总览](RESULTS_OVERVIEW.md)，
算法说明和历史 32B 表格见[项目 README](../README.md)。
