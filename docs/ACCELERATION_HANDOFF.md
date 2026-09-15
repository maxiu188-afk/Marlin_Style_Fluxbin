# 加速实验合同与恢复交接

更新：2026-09-15。最新结果为 prepared v2.1 全模型 Graph 对原始 BF16
1.381x / 1.383x，两组稳定；同轮 eager 不稳定。当前状态见
[当前交接](CURRENT_HANDOFF.md)，逐轮证据见[加速结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。
本页保存有效合同，不追加已过期的“下一步”快照。

## 固定模型与权重

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 36 层、252 个 Linear，固定 hybrid distilled step400；embedding、norm、lm_head 保留原参数。
- 表示：2 个 global bases + 2 个 sparse refinement bases，group=128、每组 8 个修正列。
- payload 目录：`/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`。
- manifest SHA256：`253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`。
- snapshot：`/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`。

上述路径是既有服务器存储位置，执行时通过环境变量指定，不写入通用代码。
大权重未下载本机；小型结果、环境、日志和验收备份留在 Git-ignored `server_results/`。
step400 test PPL 13.169788495，原质量门槛未通过；用户授权性能研究先行，精度工作暂停。

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

最后一次服务器任务已退出、无 GPU 进程，证据备份完成；用户尚未确认本实例关闭。
关闭计算实例时保留 `/workspace` 网络卷 `34au39ljvf`、模型、结果及缓存。
`/opt/fluxbin-venv` 在容器盘，重启后按
`infra/runpod/requirements-linear-a100-v1.lock` 建立本机独立环境，不跨机器复制 venv。
复用模板 torch 和兼容编译缓存，最近恢复约 40 秒；基础镜像 digest 未确认，镜像自动化暂缓。

下一次先核对 Git/存储/权重，再生成匹配源码的环境记录、跑 CUDA 检查并使用新的结果目录。
入口与命令见[运行手册](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)。本次文档整理不启动实验。
未来若只更新 scales，须重做新权重的正确性检查；性能结论仍绑定原 manifest 和执行环境。
只有格式、形状、索引、dtype、dispatch 和无数据相关分支均不变时，才可认为计算/访存规模不变。
