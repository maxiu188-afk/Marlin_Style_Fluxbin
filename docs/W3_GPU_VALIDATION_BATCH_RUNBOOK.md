# W3 GPU 验证批次：一次租卡跑完的四个作业

更新：2026-09-17。GPU 按小时租,所以把当前所有待验证项打包成一次会话。本页是
执行手册;批次尚未运行,这里没有任何结果。

入口:`scripts/run_w3_gpu_validation_batch.sh`,汇总
`scripts/summarize_w3_gpu_validation_batch.py`。

## 批次要回答什么

| # | 作业 | 回答的问题 | 约耗时 |
|---|---|---|---|
| 1 | `bank-conflict-probe` | shared-memory bank conflict 是不是 W3 kernel 的主瓶颈? | ~5 min |
| 2 | `split-sweep` | A100 上 gps / row-tile 的最优点在哪(含未测过的 gate/up gps2、R2048)? | ~15 min |
| 3 | `full-model-stock` | 同会话的非融合对照 | ~35 min |
| 4 | `full-model-fused` | RMSNorm/RoPE 融合在真机上值多少? | ~35 min |

顺序是**最便宜且信息量最大的在前**。每个作业 fail-soft:失败只记录并继续,不让
一个坏作业浪费整段租期。批次可续跑——结果 JSON 已存在的作业会跳过,所以断线只
损失当时正在跑的那一个。

## 为什么是这四个

**1. bank-conflict probe**。kernel 只跑到 cuBLAS BF16 有效带宽的 19--52%,最可能的
原因是 `lut[chunk*256 + p]` 的索引是权重字节,一个 warp 的 32 条 lane 打进 256 项表
的 32 个随机 bank(32 球 32 桶,期望最大桶约 3.4)。Nsight 在现有宿主一直是
`ERR_NVGPUCTRPERM`,拿不到硬件计数器,所以用计时替代:探针构建把 lane 0 的索引广播
给全 warp,消除 conflict 同时保持权件 load 存活。**探针构建的输出按构造就是错的**,
脚本因此只报计时、标 `timing_only`,并断言探针输出确实与真实输出不同——否则就是
`-D` 没生效、在给同一个 kernel 计时两次。

差值是**下界**:广播本身要花一条 shuffle。已有一个独立旁证:09-17 的
`fast_corrected` 每 chunk 多 2 次 LUT 查表(+67%),整模慢 9.7%,反推查表约占
kernel 关键路径 37%。探针应当与这个量级一致。

**2. split-sweep**。`configs/acceleration/w3_lut_bf16_split_candidates_v1.json` 的
46 个 cell 已经实现、已在 RTX PRO 4500 上跑过,**从没在 A100 上跑过**。它已经覆盖
gate/up 与 down 的 R=2048 和 gps 1/2/4/8。这很重要,因为 09-17 的 `observed_exact`
(gate/up gps4、down gps2)在 A100 上比 gps1 慢 5.8%——gate/up 从 384 个 block 掉到
96 个,低于 108 个 SM。中间点 gps2(192 blocks)未测,可能是最优点。

**3/4. 全模型 stock 与 fused**。融合会同时加速两个臂,所以**融合增益必须在同一会话
内对照**,否则就是跨实例比较。两者除 `fused_nonlinear_modules` 外逐字段相同(有测试
比对),但是**不同的 protocol id**,结果不能混进同一张表。

## 执行

```bash
export FLUXBIN_PYTHON=/workspace/environments/<env-id>/venv/bin/python
export FLUXBIN_SNAPSHOT_ROOT=/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export FLUXBIN_GPTQ_ROOT=/workspace/models/fluxbin/qwen3-8b-gptq-w3
export FLUXBIN_W3_LAYOUT_ROOT=/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1
export FLUXBIN_W3_LAYOUT_MANIFEST_SHA256=f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8
export TORCH_EXTENSIONS_DIR=/workspace/cache/torch-extensions/w3-batch-$(git rev-parse --short HEAD)-cu128-sm80
export TORCH_CUDA_ARCH_LIST=8.0
```

在 tmux 里:

```bash
bash scripts/run_w3_gpu_validation_batch.sh \
  /workspace/jobs/w3-validation-batch-$(date -u +%Y%m%d) \
  /workspace/results/w3-validation-batch-$(date -u +%Y%m%d)
```

`FLUXBIN_GPTQ_ROOT` 只有作业 2 需要(它要读 `decoded_bf16/layer-000.safetensors`);
若该目录不在此实例上,作业 2 会失败而其余三个照常完成。

**必须先固定解释器。** 2026-09-17 retry1 的首次启动因 quoting 选到 system Python,
在模型加载前就失败。脚本因此在跑任何作业前先验证 `FLUXBIN_PYTHON` 存在且
`torch.cuda.is_available()`,并拒绝脏 worktree。

批次还会做一次 **Nsight 复检**并写入 `nsight-recheck.log`。如果这台实例允许硬件
计数器,那么 profile kernel 的价值高于批次里的任何一项,应当在租期结束前补上。

## 结果解读

汇总写入 `<result>/batch-summary.json`。

| 观测 | 含义 |
|---|---|
| 探针 `conflict_cost_fraction_lower_bound` ≳ 0.3 | conflict 是主瓶颈,kernel 重写应当针对它(查表结构,不是 LUT 构建) |
| 探针 ≲ 0.1 | conflict 不是主因,转去查 DRAM 访存模式与 occupancy |
| sweep 中 gate/up 最优为 gps2 | 采纳为新的冻结候选,可再做一轮全模型 |
| sweep 中最优仍为 gps1 | split-G 路线到此为止,不再投入 |
| fused 的 `original_bf16` 每 token 明显低于 stock | 融合生效;按**绝对吞吐**报告,不要只报加速比 |
| fused 与 stock 的 `original_bf16` 基本相同 | torch.compile 没有在 CUDA Graph 下生效,查 capture 前的 warm-up |

作业 3/4 都会自动产出逐步 NRMSE(`stepwise_normalized_rmse`),其中对照臂的
step-0 dyn-vs-static 值正是[数值门标定](W3_NUMERICAL_GATE_CALIBRATION.md)缺的那项。

## 边界

- 探针构建数值无效,只能用于计时,永远不得进入路由或正式输出。
- fused 与 stock 是不同 protocol id,不得与 frozen structural v1 的历史数字并表。
- 作业失败表示**没有证据**,不是负面结果。
- 本批次不含 lm_head 量化、整层 torch.compile 和 SiLU-mul 融合;理由见
  [非 Linear 路径融合](NONLINEAR_FUSION.md)。
