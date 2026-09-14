# 精度暂停、加速研究与关机交接

2026-09-14 更新：先 M=1 kernel → 单个完整 block → 全模型；未来接入 vLLM，
当前仅保留 engine 接口。本地准备与首次服务器环境记录/后续镜像流程见
[M1_ACCELERATION_PREPARATION.md](M1_ACCELERATION_PREPARATION.md)。
本轮不连接服务器，不运行 GPU 实验，不构建或发布镜像。

用户已决定：完成固定 step400 test PPL 后暂停精度实验，项目主要转向加速效果研究。
最终 test PPL 13.169788495，较未蒸馏直接父版本 14.951611048 降低 11.9173%。
混合、补偿修复、蒸馏的阶段证据见 `QWEN3_8B_DISTILLED_TEST_RESULTS.md`。
这次没有运行 CUDA 性能实验，不宣称任何加速数值。

## 已持久化的性能研究起点

网络卷挂载 `/workspace`，来源 `mfs#euro.runpod.net:9421`，卷路径
`/networkvolumes/34au39ljvf`。蒸馏权重已另存并逐文件 SHA256 校验到：

```text
/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/
  manifest.json
  payloads/layer-000.safetensors ... layer-035.safetensors
  result.json / acceptance.json / provenance.json
  normalizers.json / validation_protocol.json
  test-ppl-result.json / test-ppl-acceptance.json
```

36 层 packed 权重约 2.54 GiB；源作业的权重和最终 optimizer state 也继续保留。
这个目录保存量化的 252 个 Linear，不是可直接 from_pretrained 的完整 HF checkpoint。
重建整个模型还需要已保留的原始 pinned snapshot 中的 embedding、norm、lm_head、
tokenizer/config：

```text
/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

`manifest.json` 记录全部文件校验值和原始 snapshot 依赖。step400 被固定为第一版
性能实验输入；不按未来精度结果追溯修改已测版本。

## 权重更新与加速结论

在以下条件不变时，后续仅改变 FP32 scales 通常不改变运算量、访存规模和预期性能：
量化格式、2 个 global bases + 2 个 sparse refinement bases、g128/s8、矩阵形状、
固定符号和索引、布局、dtype、batch/sequence、kernel dispatch、软硬件环境。
这也是本项目把精度迭代与加速研究分开的前提。

每次性能报告仍记录权重 manifest、kernel 和环境版本。若 kernel 有数值依赖的
跳零/剪枝/压缩分支，或者后续改变索引、位宽、稀疏度、layout、算子路径，则需
重新确认性能；不能无条件断言任意权重修改都不影响速度。只改变 scales 时也需
为新权重重做数值正确性检查，但既有固定配置的计时结论仍属于其原测量条件。

## 后续加速实验顺序（本次不启动）

1. 先冻结执行语义：输入/累积 dtype、scale 应用顺序、支持形状、数值容限及 fallback。
2. 从现有算法 packed 格式转换成单独版本的 kernel layout，不重新量化。
   CPU/reference 解码与最终算法权重一致；检查不改变符号/索引/scales。
3. Linear correctness → Linear 计时 → block → full-model；每级分别验收。
4. 固定 GPU、batch/token、warmup/repeats、同步与计时范围；说明转换/打包是否计入，
   与相同输入/输出和 dtype 的 baseline 对照。不得用 PPL 或 weight SSE 代替速度证据。
5. 暂停新增蒸馏、校准和精度参数搜索；以后恢复精度实验使用新的版本目录。

原 5%/10% PPL 门槛的数值结果仍留在报告中；用户此次明确授权性能研究先行，
不再以质量门槛未通过阻止研究性 correctness/benchmark。产品部署适用性仍需单独判断。

## 关闭和下次恢复

- 关闭计算实例时保留上述网络卷；不要删除卷或持久目录。
- `/opt/fluxbin-venv` 属于 container disk，下次需要按既有 lock 恢复；不用重装模板的 torch。
- 环境锁：`infra/runpod/requirements-linear-a100-v1.lock`，本轮 torch 2.8.0+cu128、
  transformers 5.14.1、datasets 5.0.0、safetensors 0.8.0。
- 本地已备份小型结果/验收/曲线/数据/日志/环境记录，位于
  `server_results/runpod_hybrid_pcie_2026-09-13/`，不入 Git；大权重按用户选择不下载本机。
- 下次先提供新 SSH 地址，确认网络卷挂载，再校验 manifest 和 snapshot；
  使用 Git 拉取源码，建立该机器独立环境，然后开始性能工程。
- 用户已确认计算服务器关闭；此前由代理完成关机准备，未通过工具关闭或删除实例。
  存储状态以上次关机前校验为准，下次启动重新核验。
