# 加速实验合同与恢复交接

更新：2026-09-17。QBB 最新性能结果仍为 prepared v2.1 全模型 Graph 对原始 BF16
1.381x / 1.383x，两组稳定；同轮 eager 不稳定。当前状态见
[当前交接](../CURRENT_HANDOFF.md)，逐轮证据见[加速结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。
后续质量/码率决策已转向 uniform 3-bit；本页只保存冻结 QBB 性能合同，不再作为
继续优化当前 QBB kernel 的指令。

uniform W3 的第一轮 backend 和完整模型 trial 随后已经完成：Graph 性能约为原始
BF16 的 1.49x，但 packed-vs-decoded W3 数值门失败。当前 W3 状态、误差归因和
下一步语义边界见 [W3 完整模型结果](W3_LUT_FULL_MODEL_RESULTS.md)；该结果不能
反向改写为 QBB 或 W3 accepted full-model correctness 证据。

两条 decoded-BF16 corrected 完整模型路线也已在 A100 PCIe 完成。按 raw CUDA
median，`fast_corrected` 为 1.3441x / 1.3427x，`observed_exact` 为
1.2703x / 1.2699x；两者的 correctness 均失败，formal comparison stability 也因
isolated outlier 未全通过。用户当前以加速效果为主，因此约 1.49x structural W3
继续作为性能主线，corrected 路线只保留为负面 follow-up。3.154552-bit W3 相对
BF16 的理论存储优势约 5.07x，但相对理想 W4 只有约 1.27x；bitplane decode、LUT、
split-G workspace/finish reduction 和 A100 缺少原生 INT3 MMA 会显著稀释带宽收益。

## 路线边界

RTX PRO 4500 同卡四臂结果中，GPTQ W3 g128 为 3.154552 bit/weight、PPL
11.266115，当前 QBB 为 3.138184 bit/weight、PPL 13.167910。GPTQ 只增加
0.5216% 存储而降低 1.9018 PPL，符合预先定义的 Case A。后续优先 uniform
3-bit backend / solver；prepared v2.1 作为历史 QBB 性能基线保留。详见
[W3/QBB 结果与决策](../quality/QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

## 固定模型与权重

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 36 层、252 个 Linear，固定 hybrid distilled step400；embedding、norm、lm_head 保留原参数。
- 表示：2 个 global bases + 2 个 sparse refinement bases，group=128、每组 8 个修正列。
- payload 目录：`/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`。
- manifest SHA256：`253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`。
- snapshot：`/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`。

上述路径是既有服务器存储位置，执行时通过环境变量指定，不写入通用代码。
大权重未下载本机；小型结果、环境、日志和验收备份留在 Git-ignored `server_results/`。
step400 A100 test PPL 13.169788495，原质量门槛未通过。之后的同码率质量实验已经
完成；“精度工作暂停”不再是当前状态。

## 当前执行与数值参照

当前 kernel 为 `v5_p1024/gps1`，离线符号 byte planes + A16 LUT；
列 scale 与激活组合，row scale 在分解内积外应用。转换不得重新量化或改变符号、
索引和 scales。Linear 用结构 FP64 参照；旧重建后 BF16 路径另行保留。

prepared v2.1 在真实前缀预填充后绑定固定布局、输出和 workspace，使用 StaticCache。
绑定只支持串行单请求，输出复用；权重变化前解除绑定，不作为并发 serving 接口。
同一静态注意力下，原 checked wrapper 与 prepared wrapper 必须逐位一致；
Graph/eager 和正式重复也必须逐位一致，检查有限值、KV 长度、相同 fed tokens、
252 条 packed 路由和零 decode dense fallback。不得静默回退或跳过失败。

packed/decoded BF16、动态/静态注意力的差异以原阈值计算并 report-only 记录，
不称为数值等价。首次 v2 的失败记录保留；当前配置文件名为 v2，内部协议 ID 为 v2.1。

## 性能口径

主基线是原始 BF16，decoded step400 BF16 是辅助基线。固定两个真实 prompt、
batch1、32 个来自 decoded 路径的相同 continuation token。每轮恢复前缀 KV，
后续位置逐步增长；不是随机 KV 或固定位置单 token 重放。

prepared eager / sequence Graph 都包含 embedding、全部 block、LM head、argmax，
以及 packed LUT 构建/归约；load、conversion、prefill、reset、capture、CPU 审计不计入 decode。
三模型与两组缓存同时驻留，轮换 arm/mode，8 轮预热、10 轮测量。
wall 和 CUDA event 极差/中位数均 <=5% 才发布该比较的速度比；event 仍可能包含 host 发射间隙。

Linear、block、完整模型分别报告。单 block 不作为全模型启动条件，不外推 Linear 时间和。
旧动态 KV v1 的 1 轮预热/3 轮测量/10% 稳定性结果不追溯改写。
跨 GPU、KV、驻留策略和协议的差异不作单因素因果判断。当前未接入 A8 或 vLLM。

## 恢复与后续工作

QBB 性能服务器任务已退出、无 GPU 进程，证据备份完成；其实际电源状态仍由用户确认。
后续 RTX 质量服务器也已完成关机审计，路径与哈希见
[关机记录](../operations/SERVER_SHUTDOWN_READY.md)。
关闭计算实例时保留 `/workspace` 网络卷 `34au39ljvf`、模型、结果及缓存。
`/opt/fluxbin-venv` 在容器盘，重启后按
`infra/runpod/requirements-linear-a100-v1.lock` 建立本机独立环境，不跨机器复制 venv。
复用模板 torch 和兼容编译缓存，最近恢复约 40 秒；基础镜像 digest 未确认，镜像自动化暂缓。

下一次先核对 Git/存储/权重，再生成匹配源码的环境记录、跑 CUDA 检查并使用新的结果目录。
入口与命令见[运行手册](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)。本次文档整理不启动实验。
只有后续明确要求恢复 QBB 研究时，若只更新 scales，仍须重做新权重的正确性检查；
性能结论继续绑定原 manifest 和执行环境。
只有格式、形状、索引、dtype、dispatch 和无数据相关分支均不变时，才可认为计算/访存规模不变。

若恢复 W3 性能研究，先在允许 performance counters 的 A100 上 profile structural
路线，量化 main/finish 双 kernel、32/96 个 partial split、LUT build/barrier 和非
Linear 固定时间，再决定 fusion、persistent kernel 或 Tensor Core-friendly W3 重构。
不自动重跑 corrected 路线，也不因性能优先而把 correctness 失败改写为 accepted。
