# W3 packed M=1 PPL 下一轮协议

## 目的与边界

下一轮不再从 logits NRMSE 推断语言模型质量，直接在冻结的 WikiText-2 test token
artifact 上测 PPL。协议使用同一批 146×2048 blocks、298,862 个 next-token
transitions，并且每一次模型调用只输入一个 token、携带增长中的 KV cache。因此 packed
臂的 252 个 Linear 全部走真实 M=1 W3 CUDA backend，不存在 dense prefill 替代。

三臂顺序固定为：

1. `original_bf16`：融合 RMSNorm/RoPE 的原始 BF16 权重；
2. `decoded_w3_bf16`：融合 RMSNorm/RoPE、W3 权重解码为 dense BF16；
3. `packed_w3_inline`：相同融合路径、structural packed W3 M=1 kernel。

`packed - decoded` 是 backend 累计数值影响，`decoded - original` 是 W3 量化质量影响。
历史 dense、`use_cache=False` 的 BF16/W3 PPL 仅作背景，不作为本轮 acceptance anchor。
本轮没有人为 PPL 阈值；只冻结数据、覆盖、有限值和 provenance，结果完成后人工评审
effect size。PPL 任务不产生性能结论，也不自动启动 kernel 修改。

## 节省启动时间

CUDA extension 使用内容寻址缓存：

```bash
export FLUXBIN_EXTENSION_CACHE_ROOT=/workspace/cache/torch-extensions/fluxbin-content
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
export TRITON_CACHE_DIR=/workspace/cache/triton
export TORCH_CUDA_ARCH_LIST=8.0
```

缓存键包含 `.cu` 内容、编译 flags、Torch/CUDA、C++ ABI、Python 和 SM 架构，不包含
Git commit。启动脚本先预热 R256/R512/R1024 并保存包含 `.so` SHA256 的 manifest；
warm cache 时只进行 Ninja/manifest 校验。不要清理持久卷上的该目录。
RMSNorm 的 Inductor/Triton 产物也使用持久缓存目录，避免每个新实例重复生成相同融合
kernel。

## 启动

服务器环境准备完成后，在独立 tmux 会话中运行：

```bash
export FLUXBIN_PYTHON=/workspace/environments/<env-id>/venv/bin/python
export FLUXBIN_IMAGE_REFERENCE=runpod-default-unresolved

bash scripts/run_qwen3_8b_w3_packed_m1_ppl_job.sh \
  /workspace \
  /workspace/jobs/qwen3-8b-w3-packed-m1-ppl-v1 \
  /workspace/results/qwen3-8b-w3-packed-m1-ppl-v1
```

每个 arm 完成后独立写入 `results/.../arms/<arm>.json`。若实例中断，用相同三个参数
重跑，启动脚本自动添加 `--resume`；runner 会先验证 config、environment、layout、
tokens 和 source hashes，只有 provenance 完全一致才跳过已经完成的 arm。

完整数据规模下预计是小时级质量任务。不要用小样本 PPL 替代正式结果，也不要在任务
运行时同时跑 CUDA 测试或性能 benchmark。
