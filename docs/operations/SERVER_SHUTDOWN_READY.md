# 服务器关机准备与恢复记录

## 2026-09-17：hierarchical-scale W2 端点实验已关机

质量服务器 `213.173.105.10:49356` 上的 A100 80GB PCIe 端点任务已
退出 0。关机前已核验结构化结果、job-produced hashes、finite metrics、
36 层/252 Linear 覆盖、无剩余 GPU/experiment 进程，并执行 `sync`。
用户已确认自行关闭服务器；本轮不再重连推断电源状态。

- 执行源码：`457008a`。
- 最终作业：`/workspace/jobs/hierarchical-w2-endpoints-20260917T104245Z/`。
- 结果根目录：`/workspace/results/qwen3-8b-hierarchical-w2-v1/`。
- 有效端点：H2.50 PPL `27.9328351515`，H2.875 PPL `27.8989296368`；
  同轮 W3 PPL `11.2668082541`。
- H2.625 的 10 个 partial layers 是中止现场，H2.75 未启动；不得将它们
  写成已完成结果。
- 持久卷：`34au39ljvf`，挂载 `/workspace`；已保留，未删除。
- 小型原始 JSON 在关机前未复制到 Mac；权威原始产物仍位于网络卷。

结论与恢复边界见
[hierarchical W2 结果与运行手册](../quality/QWEN3_8B_HIERARCHICAL_W2_RUNBOOK.md)。

## 2026-09-16：W3 完整模型与误差诊断已收尾

W3 性能服务器 `213.173.105.10:43680` 为 A100 80GB PCIe。正式 full-model
任务和两项只读诊断均已结束；最后检查无 NVIDIA compute 进程或 tmux 会话，远端
Git 工作区干净且为 `fc76b75802d57e238eaf258ab85f3803588e92c8`。

- 正式状态：`completed_with_backend_numerical_differences`；Graph 性能稳定，
  packed-vs-decoded W3 correctness gate 未通过。
- 远端结果：`/workspace/results/qwen3-8b-w3-full-m1-v1/`。
- 远端作业记录：`/workspace/jobs/qwen3-8b-w3-full-m1-v1/`。
- W3 layout：`/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1/`；manifest
  SHA256 `f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8`。
- 正式结果 SHA256：
  `c68811aa065b34d497f3113e0bc873684624ee0c8fc8df199998a7e53856f423`。
- environment SHA256：
  `0e75d87e3fe7c2de4525cd0b3349656dae2088ad6778a017c3c1be11f710d558`。
- 逐层归因 / 算术分解 SHA256：
  `8620846807f7214912c752aeb79d6011f61892d71dd1c22a81e96ca96d87cbe6` /
  `f1767d0e7e673a31923573eafa9c38673c3698fec61abc1e653fe2a386a64a85`。
- 本地私有备份：`server_results/runpod_w3_full_m1_a100_pcie_2026-09-16/`。

可以关闭该计算实例，但必须保留 `/workspace` 网络卷。正式结果、环境、日志和诊断
JSON 已下载并核对哈希；大模型和 W3 artifact 继续只保存在网络卷。完整结论见
[W3 完整模型结果](../performance/W3_LUT_FULL_MODEL_RESULTS.md)。

## 2026-09-16：RTX PRO 4500 质量实验已收尾

质量服务器 `213.173.109.240:44534` 上的 W3/QBB 四臂 PPL 作业退出 0，状态为
`completed_cross_device_pending_review`。每个 arm 均覆盖 36 layers、252 Linears、
146 blocks 和 298,862 positions；最终 Case A 决策见
[W3/QBB 状态页](../quality/QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

- 执行源码：`f979f6b3f64aa7ccabd2d96cc5e953a83e2d7080`；结果整理与关机说明另行
  提交到 `codex/runpod-bootstrap`。
- summary JSON SHA256：
  `de5e12fcd7c1db5da1952717dd15a01c9c230b34a3d2b254e64d0fc05e88e8cd`。
- 远端结果：`/workspace/artifacts/qwen3_8b_w3_rate_distortion/`。
- 远端 closeout：
  `/workspace/jobs/qwen3-8b-w3-rate-distortion-v1/closeout-rtx4500-20260916/`。
- closeout archive SHA256：
  `812ba530f9fb4a41f97776c952c5eb5ce4f5dfef8e34fb776729a03abe49bfe7`。
- 本地私有备份：`server_results/runpod_w3_rate_distortion_rtx4500_2026-09-16/`；
  归档已解包并复核，共 48 个小型 evidence 文件。模型和低比特权重未下载。
- 最终检查已执行 `sync`；无 tmux、GPU compute 或实验进程，Git 工作树干净。

可以关闭该计算实例，但必须保留网络卷 `34au39ljvf`。`/opt/fluxbin-venv` 位于
容器盘，可随实例丢弃；恢复使用冻结 lock、持久模型/snapshot 和 Git 提交。

2026-09-15 prepared SXM4 trial on `213.173.102.5:11028`: completed; retry exit 0,
Graph full-model speed 1.381x/1.383x original BF16 (stable), eager unstable.
Both attempts, environment/tests/job logs and final JSON were downloaded and hash-verified.
No GPU experiment processes remain. Compute instance may be stopped; retain network volume
`34au39ljvf` and its models/cache/results. See `QWEN3_8B_M1_LINEAR_RESULTS.md`.

## 2026-09-15：inner compute PCIe 实验已收尾

106 项测试和 24 个候选 trial 完成；v5_p1024/gps1 完整模型已结束，未加速。
全部小型证据已打包下载，归档 SHA256
`28d4476d8bb9183032725788b75e4b1eb450293923f51e6c34f197100de82647`
本地/远端一致。最后检查无 GPU compute 进程、无 tmux 会话。可以关闭计算
实例，保留 `/workspace` 持久卷；未代用户执行关机或删除。详见当前结果文档。

## 2026-09-14：SXM4 实例关机前检查完成

状态：可以由用户关闭计算实例，尚未确认实际关闭；代理未执行关机或删除操作。

- 无 tmux 会话、相关实验进程或 NVIDIA compute 进程。
- setup/candidates/block/report-only full-model 退出 0；首轮 strict full-model 的退出 1
  是已记录的数值 gate 失败，原始现场保留。
- `/workspace` 仍挂载原网络卷 `34au39ljvf`；保留该卷及其模型、snapshot、缓存和结果。
- step400 manifest SHA256 仍为 `253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`。
  模型目录存在，本轮关机检查没有再次扫描全部大权重；全模型运行已做完整输入 hash 校验。
- 服务器 Git 工作区干净，运行代码 revision 为 `d1bcd6d`；之后发布的是本地结果文档。
- 全部本次小型结果/环境/日志/启动脚本共 68 个文件已归档并下载本机，远端/本地 SHA256
  一致：`6b08d387c19bf1e50d6a03dd865e092e41955160cfc132832b5476d19d49b1e4`。
  本地归档：`server_results/runpod_m1_sxm4_2026-09-14/m1-sxm4-20260914-evidence-v1.tar.gz`。
  远端归档：`/workspace/results/m1-sxm4-20260914-evidence-v1.tar.gz`。
- `/opt/fluxbin-venv` 在容器磁盘，不应依赖下次保留。恢复使用已记录的软件版本与
  持久缓存；trial runtime 的 OMP/MKL 线程数均为 1。

完整模型结果已保存：packed 相对原始 BF16 为 0.9332× / 0.9562×，尚无加速；
按用户要求记录数值差异并完成计时。下次不自动重跑，先按新任务明确优化方向。

## 2026-09-13：历史实例已关闭

2026-09-13：用户已确认关闭计算服务器。以下存储与进程状态来自关机前核查，
本次未重新连接已关闭实例；下次启动先核验原网络卷和 manifest。

关机前核查完成：

- 蒸馏和 test PPL 均已结束，test 退出码 0；最后查询没有 NVIDIA compute 进程。
- 蒸馏后 step400 test PPL **13.169788495**，比未蒸馏直接父版本 14.951611048 降低 **11.92%**。
- 36 层蒸馏权重已另存到真实网络持久盘，逐文件 SHA256 校验通过。
- test 结果/验收、训练来源和模型 manifest 已随持久权重保存；本地小型备份已验证。
- 远端 Git checkout 干净；本次没有运行加速实验，没有代为关闭/删除实例。

**关闭计算实例时保留网络卷，不删除卷。**

持久卷：`/workspace`，挂载来源包含 `/networkvolumes/34au39ljvf`。
关键模型目录（约 2.54 GiB）：

```text
/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/
```

manifest SHA256：`253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`。

此目录保存全部量化 Linear 和记录；恢复完整模型仍需保留原始 snapshot 的
embedding/norm/lm_head/tokenizer/config：

```text
/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

原始训练输出、optimizer state、旧版本权重也仍在 `/workspace`，未删除。
本地备份目录：`server_results/runpod_hybrid_pcie_2026-09-13/`。
最新关机小备份：`distilled-test/distilled-test-closeout-small.tar.gz`，含 test 结果、
验收、日志、持久模型清单和环境记录；大权重按此前决定只保留网络盘，不下载本机。

`/opt/fluxbin-venv` 是 container disk 环境，下次按 lock 恢复即可；不用重新下载模型。
当时的下一步是提供新 SSH、挂载原网络卷并核验 manifest；“精度实验暂停”是
2026-09-13 的历史快照，已被 2026-09-16 W3/QBB Case A 决策取代。
