#!/usr/bin/env bash
set -euo pipefail

persist_root="${PERSIST_ROOT:-/workspace}"
environment_id="${FLUXBIN_ENV_ID:-runpod-v1}"
cache_arch="${FLUXBIN_CACHE_ARCH:-}"

if [[ -z "${cache_arch}" ]]; then
    cache_arch="$(python - <<'PY'
try:
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"sm{major}{minor}")
    else:
        print("no-gpu")
except Exception:
    print("unknown-gpu")
PY
)"
fi

cache_namespace="${environment_id}-${cache_arch}"

cache_root="${persist_root}/cache"
mkdir -p \
    "${cache_root}/huggingface" \
    "${cache_root}/torch-extensions/${cache_namespace}" \
    "${cache_root}/triton/${cache_namespace}" \
    "${cache_root}/pip" \
    "${persist_root}/artifacts" \
    "${persist_root}/datasets" \
    "${persist_root}/models" \
    "${persist_root}/repos" \
    "${persist_root}/results"

export HF_HOME="${cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${cache_root}/huggingface/hub"
export HF_DATASETS_CACHE="${cache_root}/huggingface/datasets"
export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions/${cache_namespace}"
export TRITON_CACHE_DIR="${cache_root}/triton/${cache_namespace}"
export PIP_CACHE_DIR="${cache_root}/pip"

exec "$@"
