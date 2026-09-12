# syntax=docker/dockerfile:1

# This documented RunPod image is a reproducible bootstrap baseline. Before the
# first accepted Qwen3-8B run, pin the image by digest and record it in the
# experiment manifest. Build for linux/amd64 when the build host is Apple
# Silicon.
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PERSIST_ROOT=/workspace \
    FLUXBIN_ENV_ID=cu1281-torch280-v1 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0"

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        jq \
        ninja-build \
        openssh-client \
        rsync \
        tmux && \
    rm -rf /var/lib/apt/lists/*

COPY infra/runpod/requirements.lock /opt/fluxbin/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /opt/fluxbin/requirements.lock

COPY infra/runpod/entrypoint.sh /usr/local/bin/fluxbin-entrypoint
RUN chmod 0755 /usr/local/bin/fluxbin-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/fluxbin-entrypoint"]
CMD ["sleep", "infinity"]
