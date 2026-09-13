# Qwen3-8B 修复版 hybrid 全模型权重验收

2026-09-13：**passed（产物完整性、固定索引与 BF16 重构）**。
修复版 PPL 随后测得 14.951611，执行有效但质量门槛未通过；见 [PPL 验收](QWEN3_8B_CONDITIONED_PPL_RESULTS.md)。

## 执行与验收

| 项目 | 结果 |
|---|---:|
| 层数 / Linear 数 | 36 / 252 |
| 量化参数数 | 6,945,767,424 |
| 退出码 | 0 |
| 量化耗时 | 1960.51 秒（32.68 分钟） |
| 峰值 torch allocated 显存 | 11.603 GiB |
| payload 张量数 | 1,764 |
| payload 总字节 | 2,724,825,888（2.538 GiB） |
| tensor-data bits/weight | 3.13818359375 |
| 重算 SSE 最大相对差异 | 3.2879e-16 |

服务器为 NVIDIA A100 80GB PCIe。执行版本
`e5f3861c22cd99bbba5cf4bb7bfdf7df3b183de5`，任务
`qwen3-8b-hybrid-conditioned-v1`。日志完整，36 层均提交并完成传播；
无断点恢复，未覆盖历史权重。量化前本地与服务器均通过 59 项测试。

独立 CPU 审计耗时 179.89 秒，检查了：

- 源码 manifest、resolved config、结果及所有层 metadata/payload 哈希。
- 36 层 / 252 Linear / 6,945,767,424 参数及完整目录、张量清单。
- 1,764 个张量的哈希、数据类型、有限性，payload 格式及回读记录。
- 全部目标 BF16 权重与原始 snapshot 的哈希匹配。
- 252 个 Linear 的 payload 解码后 SSE 与原记录一致；汇总重新计算一致。
- 每个 Linear 的旧拟合列索引哈希等于修复版列索引哈希，且等于保存的索引。
- 稀疏修正未改动未选中列；所有 Hessian 记录均为 524,288 个输入行。

这里的 CPU 审计重新执行 packed 解码和计分，未重新拟合或独立重放全模型 Hessian。

## 数值结果与边界

普通权重 SSE 为 **693294.065948**，relative Frobenius 为 **0.375446280**。
相对旧 hybrid SSE 637440.223930，增加约 **8.76%**。
补偿修复优化输入加权误差，普通权重 SSE 上升不等同于 PPL 上升或下降。

逐层 FP32 Hessian 诊断汇总为 56766.329962。新旧量化前缀的输入不同，且该指标
未采用小 probe 的独立 FP64 重放，因此不把其变化当作严格同输入收益或 PPL 证据。
下一步应运行原 WT2 评估协议，与 BF16 9.724945、旧 hybrid 16.142104 对照。
本次验收未启动 PPL、pure、蒸馏或后端测试。

## 可追溯记录

- resolved config SHA256：`d984b89de72a41deba055135fa0e58f3f7c10d1873a8637d20a350003b1f0ba4`。
- result SHA256：`e00371bbfa6091ea5f927eb61fb3808d1ba96c25441ed373752f522026c343d7`。
- 审计脚本 SHA256：`226f99fee22c11663420ff03af6839654e9dc222f594b0e168e4945c035ecab7`。
- 服务器 checkout：`results/qwen3-8b-full-hybrid-conditioned-v1/acceptance.json`、
  `portable-payload-manifest.json`；权重仍在独立的同名 artifacts 目录。
- 本地小型备份：`server_results/runpod_hybrid_pcie_2026-09-13/full-conditioned/conditioned-acceptance-small.tar.gz`。
  包含结果、验收、配置、36 层 metadata、完整日志和审计脚本；不包含模型权重，不入 Git。
