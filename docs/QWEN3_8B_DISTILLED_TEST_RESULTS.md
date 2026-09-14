# 固定 step400 蒸馏权重：WT2 test PPL

2026-09-13：执行验收通过。在本次 8B 配置和固定 test 协议下，蒸馏进一步改善 PPL。
按用户最新决定，本轮后暂停精度实验，后续以固定权重开展加速研究。

| 权重版本 | WT2 test PPL |
|---|---:|
| BF16（本轮及历史精确一致） | 9.724944981 |
| pure（历史） | 1149.470624707 |
| 原 hybrid（历史） | 16.142104443 |
| 修复补偿 hybrid、未蒸馏（历史） | 14.951611048 |
| 修复补偿 hybrid、蒸馏 step400（本轮） | **13.169788495** |

蒸馏相对直接父版本降低 **11.9173%**，绝对下降 1.781823；补偿修复与蒸馏合计
相对原 hybrid 降低 **18.4134%**。这些结果支持本项目当前阶段结论：混合和蒸馏
可以优化这套配置下的 PPL。不能把这些幅度推广到未测配置。

本轮同轮测 BF16 与最终 step400，历史版本来自相同 token/block 协议；
旧 hybrid 未在本轮重新测量。没有按 test 挑选训练 checkpoint，也没有测试 step300。

## 验收范围

- 同一 pinned Qwen3-8B snapshot；66 项本地/服务器测试通过后执行。
- A100 80GB PCIe，SDPA、BF16 logits、FP32 CE、batch 1、禁用 cache/TF32。
- 两组各 146 个 2048-token 完整块，298862 个预测位置，无非有限块。
- 36 层、252 Linear、1764 个张量均由已验收 step400 payload 装载。
- 退出码 0；源码、输入、权重、验收绑定和日志/progress 均复核通过。
- 复算 total NLL→mean NLL→PPL；未独立重放前向和逐 token logits。
- 蒸馏后 mean NLL 2.577925456003322，total NLL 770443.9576320648。

相对 BF16 仍高 35.4228%，原 ≤5% 质量门槛没有通过，旧数值判定保留。
用户已明确把当前研究重点改为加速实验，因此不再把继续优化精度作为性能研究的
前置条件；这不是声称模型已经满足原产品质量门槛。

## 固定版本

- 测试源码：`8847049a1d4a1209719e84dd5dee4f6b98fa3081`。
- result SHA256：`1b1b4353b200a0a0daa357c3e12d3a35544e6fc2f9d16c427e958644baf750ef`。
- config SHA256：`866bb246ab428de39a3f5acd39ce7b30a1e45546c52e8e78d1b26073a75c6fb2`。
- 本轮直接使用持久模型目录中的 payload，没有临时重拟合权重。
- 持久目录：`/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`。
- 本地证据：`server_results/runpod_hybrid_pcie_2026-09-13/distilled-test/distilled-test-closeout-small.tar.gz`。
- 后续实验/关机交接：`ACCELERATION_HANDOFF.md`。
