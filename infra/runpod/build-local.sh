#!/usr/bin/env bash
set -euo pipefail

# Build the NVIDIA deployment image on either Intel or Apple Silicon hosts.
# This only builds/loads the image; it never provisions a Pod or runs a model.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker_bin="${DOCKER_BIN:-docker}"
image_tag="${FLUXBIN_IMAGE_TAG:-marlin-style-fluxbin:cu1281-torch280-v1}"
output_dir="${FLUXBIN_BUILD_OUTPUT:-${repo_root}/tmp/runpod-build}"

if ! command -v "${docker_bin}" >/dev/null 2>&1; then
    echo "Docker CLI unavailable. Install and start Docker Desktop first." >&2
    exit 1
fi
"${docker_bin}" info >/dev/null
"${docker_bin}" buildx version
mkdir -p "${output_dir}"

# No mode=max cache export: avoid duplicating the large CUDA base layers.
# Loading locally is not registry publication or acceptance of a GPU runtime.
"${docker_bin}" buildx build \
    --platform linux/amd64 \
    --file "${repo_root}/infra/runpod/Dockerfile" \
    --tag "${image_tag}" \
    --metadata-file "${output_dir}/metadata.json" \
    --progress plain \
    --load \
    "${repo_root}" 2>&1 | tee "${output_dir}/build.log"

"${docker_bin}" image inspect "${image_tag}" > "${output_dir}/image-inspect.json"
echo "Built ${image_tag}; metadata and logs: ${output_dir}"
echo "Registry digest and NVIDIA GPU preflight remain required before experiments."
