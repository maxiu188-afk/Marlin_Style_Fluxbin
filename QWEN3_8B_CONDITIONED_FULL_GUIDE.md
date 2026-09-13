# 8B hybrid 补偿修复：全模型阶段

本阶段承接已验收的三个 Linear 小对照，扩展到 36 层 / 252 个 Linear。
只运行修复版 hybrid，沿用 C4 256×2048、seed 20260902、g128/s8、50 次 ALS、
1% damping 和逐层校准顺序；蒸馏尚未开始。

## 受控改动

每个 Linear 先对当前目标 W 和 H 执行旧 hybrid，取得列索引，再把相同索引
交给 `quantize_hybrid_conditioned_v1`。两次拟合共享 W/H 和求解器参数。
旧拟合只用于确定列索引，最终保存并传播修复版权重。

前层量化结果变化会自然改变后层输入，因此后层索引不保证等于历史全模型索引；
其选择流程保持旧算法。本阶段不改变层内 Hessian 捕获顺序或迭代预算。
记录旧、新索引哈希并检查一致，验证 packed BF16 回读和断点恢复路径。

冻结模板：`configs/experiments/qwen3_8b_full_hybrid_conditioned_v1.json`。
模板绑定已验收 probe 的结果文件哈希，拒绝 pure、未通过的 probe 或配置漂移。
执行前将旧 full.config 的已验证输入哈希绑定至新模板，生成独立 resolved config。
新产物：`artifacts/qwen3-8b-full-hybrid-conditioned-v1/hybrid_s8/`。
旧模型、旧权重、失败 probe 和已验收 probe 均保留。

## 验收顺序

1. 输入/运行时预检通过后量化，每层原子保存，可从完整层边界恢复。
2. 完成后检查 36 层、252 个 Linear、6945767424 个参数、全部 payload 哈希、
   索引一致和 BF16 重构。全模型旧式 FP32 Hessian 指标仅作近似诊断，
   不用作小 probe 的 FP64 一致性门槛。
3. 权重验收后再绑定新权重执行原 WT2 PPL 协议，与 BF16 9.724945 和旧 hybrid
   16.142104 比较。全模型输出误差或小 Linear 降幅不能代替 PPL。
4. PPL/后端/蒸馏均不由量化脚本自动触发。

本地 59 项测试通过，包括新 full 路由与 probe 的数值一致性、固定列与 packed
回读，以及 pure/缺失 probe 拒绝检查。服务器验证和提交状态见后续记录。
