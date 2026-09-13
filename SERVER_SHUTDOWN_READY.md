# 本次服务器可以关闭计算实例

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
下一次先提供新 SSH 地址、挂载原网络卷并核验 manifest。精度实验暂停，加速工作以后再做。
