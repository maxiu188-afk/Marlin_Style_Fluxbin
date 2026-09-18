# 修复版 hybrid PPL 验收

2026-09-13：**执行验收通过，质量门槛未通过**。

| 模型 | WT2 PPL | 相对同轮 BF16 |
|---|---:|---:|
| BF16（本轮） | 9.724944981 | — |
| 旧 hybrid（历史） | 16.142104443 | +65.99% |
| 修复版 hybrid（本轮） | 14.951611048 | +53.74% |

修复版较历史 hybrid PPL 下降 **7.3751%**（绝对下降 1.19049）。
补偿修复有实际收益，但不足以解释或消除全部质量差距。
保留原质量门槛：相对 BF16 ≤5% 才通过，>10% 继续阻止部署。
没有启动蒸馏、pure 重跑或后端测试。

## 执行和计分

- A100 80GB PCIe；执行源码 `3a3ef5c2aeee260b3b7522171b37536b7fb4d3f5`。
- 作业 `qwen3-8b-conditioned-ppl-v1`；退出码 0，总耗时 109.78 秒。
- 两组各 146 个完整 2048-token 块，298862 次预测，无非有限块。
- BF16 logits、FP32 交叉熵、batch 1、SDPA、禁用 cache/TF32。
- BF16 mean NLL 2.274694231976；hybrid mean NLL 2.704819056447。
- 完整装载 36 层 / 252 Linear / 6945767424 个量化参数、1764 个内部张量。

旧 hybrid 在 A100 SXM4 上测得，本轮 BF16 与修复版 hybrid 在 PCIe 上同轮测量。
历史与本轮协议/数据哈希一致，BF16 PPL 精确复现；旧 hybrid 未在本轮重测。
因此 7.3751% 是历史对照改善，不描述成同轮三组实验。

## 验收检查

重新运行预检：原始 snapshot、量化 payload、接受的权重审计、token artifact、
block 哈希全部通过；源码文件哈希与记录一致。日志完整，progress 与最终结果一致。
复核各组 total NLL / 298862 = mean NLL，exp(mean NLL) = PPL；重新计算质量判定一致。
验收不重复前向推理或保存/重放逐 token logits。

- result SHA256：`5258c7c011be5b6f88a78fc3b5af22d81e1d50c11f1f0593dea21951cab2ce8b`。
- config SHA256：`6458ac82dd4961c13711287f5ebf3c181e10174b691303ef2f075bb1c701fc28`。
- log SHA256：`6eca86ca76c45562e2af112b249bbe49daae1a838023973c8b2eff1e087ef017`。
- 审计脚本 SHA256：`814e8ac7b193db335f889d60ad70b332af9b65a80b7f82f7633d12e0115253fb`。
- 服务器：`results/qwen3-8b-conditioned-ppl-v1/{result.json,acceptance.json}`。
- 本地证据：`server_results/runpod_hybrid_pcie_2026-09-13/conditioned-ppl/conditioned-ppl-evidence.tar.gz`。

下一步仍属于算法质量诊断；不能根据本轮结果放行部署。
