// GPTQ W3 g128 M=1 inline activation-LUT kernel.
// The offline layout is [G,3,O,16] uint8 with an inverted high bit plane.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "lut8_unsigned.cuh"

#ifndef W3_LUT_ROWS
#define W3_LUT_ROWS 256
#endif

constexpr int kThreads = 256;
constexpr int kRows = W3_LUT_ROWS;
constexpr int kRowsPerThread = kRows / kThreads;
constexpr int kChunks = 16;
constexpr int kTables = 16;
static_assert(kRows == 256 || kRows == 512 || kRows == 1024 || kRows == 2048,
              "W3_LUT_ROWS must be 256, 512, 1024 or 2048");

template <class T, bool DenseBf16WeightSemantics, bool SingleSplit>
__global__ void w3_lut_inline_main(const T* __restrict__ x,
                                   const uint8_t* __restrict__ planes,
                                   const at::BFloat16* __restrict__ scales,
                                   const int16_t* __restrict__ perm,
                                   float* __restrict__ partial,
                                   T* __restrict__ out,
                                   int O, int G, int groups_per_split) {
  __shared__ float lut[kTables * 256];
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int first = int(blockIdx.x) * kRows + thread;
  float result[kRowsPerThread] = {};

  const int group_begin = int(blockIdx.y) * groups_per_split;
  const int group_end = min(G, group_begin + groups_per_split);
  for (int group = group_begin; group < group_end; ++group) {
    // Eight warps build sixteen tables. The activation gather implements the
    // exact desc_act canonicalization; weights and x use the same sort index.
    for (int chunk = warp; chunk < kChunks; chunk += kThreads / 32) {
      float values[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int sorted_k = group * 128 + chunk * 8 + j;
        values[j] = float(x[int(perm[sorted_k])]);
      }
      fluxbin_w3_lut::build_lane_unsigned(lut + chunk * 256, values, lane);
    }
    __syncthreads();

    #pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int output = first + row * kThreads;
      if (output >= O) continue;

      uint4 packed[3];
      #pragma unroll
      for (int plane = 0; plane < 3; ++plane) {
        const uint8_t* address = planes + (((group * 3 + plane) * O + output) * 16);
        packed[plane] = *reinterpret_cast<const uint4*>(address);
      }
      const unsigned words[3][4] = {
          {packed[0].x, packed[0].y, packed[0].z, packed[0].w},
          {packed[1].x, packed[1].y, packed[1].z, packed[1].w},
          {packed[2].x, packed[2].y, packed[2].z, packed[2].w},
      };
      float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;
      float rounding_correction_dot = 0.f;
      #pragma unroll
      for (int chunk = 0; chunk < kChunks; ++chunk) {
        const int word = chunk >> 2;
        const int shift = 8 * (chunk & 3);
        const unsigned p0 = (words[0][word] >> shift) & 0xffu;
        const unsigned p1 = (words[1][word] >> shift) & 0xffu;
        // Plane 2 is stored inverted, so p2=0 means the logical high bit is 1.
        const unsigned p2 = (words[2][word] >> shift) & 0xffu;
        accum0 += lut[chunk * 256 + p0];
        accum1 += lut[chunk * 256 + p1];
        accum2 += lut[chunk * 256 + p2];
        if constexpr (DenseBf16WeightSemantics) {
          const unsigned plus_three = p0 & p1 & (~p2 & 0xffu);
          const unsigned minus_three = p0 & (~p1 & 0xffu) & p2;
          rounding_correction_dot += lut[chunk * 256 + plus_three]
                                   - lut[chunk * 256 + minus_three];
        }
      }
      const float integer_dot = accum0 + 2.f * accum1 - 4.f * accum2;
      const float scale = float(scales[group * O + output]);
      float value = integer_dot * scale;
      if constexpr (DenseBf16WeightSemantics) {
        // BF16 multiplication by powers of two is exact. For signed W3 codes
        // {-4,...,3}, only +/-3 can require an additional weight-rounding
        // correction relative to scale * integer_dot.
        const float rounded_three = __bfloat162float(__float2bfloat16_rn(3.f * scale));
        const float delta = rounded_three - 3.f * scale;
        value += rounding_correction_dot * delta;
      }
      result[row] += value;
    }
    if (group + 1 < group_end)
      __syncthreads();  // all readers finish before the next group overwrites LUT
  }

  #pragma unroll
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int output = first + row * kThreads;
    if (output < O) {
      if constexpr (SingleSplit)
        out[output] = T(result[row]);
      else
        partial[blockIdx.y * O + output] = result[row];
    }
  }
}

template <class T>
__global__ void finish_m1(const float* __restrict__ partial,
                          T* __restrict__ out, int O, int G) {
  const int output = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= O) return;
  float value = 0.f;
  for (int group = 0; group < G; ++group)
    value += partial[group * O + output];
  out[output] = T(value);
}

template <class T, bool DenseBf16WeightSemantics>
void launch_main(const T* x, const uint8_t* planes, const at::BFloat16* scales,
                 const int16_t* perm, float* partial, int O, int G,
                 cudaStream_t stream) {
  const dim3 grid((O + kRows - 1) / kRows, G);
  w3_lut_inline_main<T, DenseBf16WeightSemantics, false>
      <<<grid, kThreads, 0, stream>>>(
          x, planes, scales, perm, partial, nullptr, O, G, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T>
void launch_finish(const float* partial, T* out, int O, int G,
                   cudaStream_t stream) {
  finish_m1<T><<<(O + 255) / 256, 256, 0, stream>>>(partial, out, O, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, bool DenseBf16WeightSemantics>
void launch_split(const T* x, const uint8_t* planes,
                  const at::BFloat16* scales, const int16_t* perm,
                  float* partial, T* out, int O, int G, int groups_per_split,
                  cudaStream_t stream) {
  const int splits = (G + groups_per_split - 1) / groups_per_split;
  const dim3 grid((O + kRows - 1) / kRows, splits);
  if (splits == 1) {
    w3_lut_inline_main<T, DenseBf16WeightSemantics, true>
        <<<grid, kThreads, 0, stream>>>(
            x, planes, scales, perm, partial, out, O, G, groups_per_split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    w3_lut_inline_main<T, DenseBf16WeightSemantics, false>
        <<<grid, kThreads, 0, stream>>>(
            x, planes, scales, perm, partial, out, O, G, groups_per_split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_finish(partial, out, O, splits, stream);
  }
}

void validate_common(const torch::Tensor& x, const torch::Tensor& planes,
                     const torch::Tensor& scales, const torch::Tensor& perm,
                     const torch::Tensor& partial,
                     int64_t expected_splits = -1) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(0) == 1,
              "CUDA x [1,K] required");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "SM80 or newer required");
  for (const auto& tensor : {planes, scales, perm, partial})
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device() && tensor.is_contiguous(),
                "all tensors must be contiguous on the input CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "x must be FP16 or BF16");
  TORCH_CHECK(planes.scalar_type() == at::kByte && planes.dim() == 4 &&
              planes.size(1) == 3 && planes.size(3) == 16,
              "planes must be uint8 [G,3,O,16]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(planes.data_ptr()) % 16 == 0,
              "planes must have 16-byte aligned storage");
  const int64_t G = planes.size(0), O = planes.size(2), K = G * 128;
  TORCH_CHECK(G > 0 && G <= 255 && O > 0 && O <= 65536,
              "dimensions outside v1 bounds");
  TORCH_CHECK(O % 32 == 0, "v1 requires O divisible by 32 for aligned plane rows");
  TORCH_CHECK(x.size(1) == K, "K mismatch");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16 &&
              scales.sizes() == at::IntArrayRef({G, O}),
              "deployment scales must be BF16 [G,O]");
  TORCH_CHECK(perm.scalar_type() == at::kShort && perm.dim() == 1 && perm.numel() == K,
              "perm must be int16 [K]");
  if (expected_splits < 0) expected_splits = G;
  TORCH_CHECK(partial.scalar_type() == at::kFloat &&
              partial.sizes() == at::IntArrayRef({expected_splits, O}),
              "partial has wrong FP32 [splits,O] shape");
  for (const auto& tensor : {x, planes, scales, perm})
    TORCH_CHECK(!partial.is_alias_of(tensor), "partial must not alias an input");
}

template <bool DenseBf16WeightSemantics>
void inline_main_impl(torch::Tensor x, torch::Tensor planes, torch::Tensor scales,
                      torch::Tensor perm, torch::Tensor partial) {
  validate_common(x, planes, scales, perm, partial);
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int O = planes.size(2), G = planes.size(0);
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_w3_lut_inline_main",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      launch_main<scalar_t, DenseBf16WeightSemantics>(
          x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
          scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
          partial.data_ptr<float>(), O, G, stream);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      launch_main<scalar_t, DenseBf16WeightSemantics>(
          x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
          scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
          partial.data_ptr<float>(), O, G, stream);
    })
  );
}

void inline_main(torch::Tensor x, torch::Tensor planes, torch::Tensor scales,
                 torch::Tensor perm, torch::Tensor partial) {
  inline_main_impl<false>(x, planes, scales, perm, partial);
}

void inline_main_bf16_weight(torch::Tensor x, torch::Tensor planes,
                             torch::Tensor scales, torch::Tensor perm,
                             torch::Tensor partial) {
  inline_main_impl<true>(x, planes, scales, perm, partial);
}

void finish(torch::Tensor partial, torch::Tensor out) {
  TORCH_CHECK(partial.is_cuda() && out.is_cuda() && partial.device() == out.device() &&
              partial.is_contiguous() && out.is_contiguous(),
              "partial/out must be contiguous on one CUDA device");
  TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.dim() == 2,
              "partial must be FP32 [G,O]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1 && out.size(1) == partial.size(1),
              "out must be [1,O]");
  TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16,
              "out must be FP16 or BF16");
  TORCH_CHECK(!out.is_alias_of(partial), "out and partial must not alias");
  c10::cuda::CUDAGuard guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int O = partial.size(1), G = partial.size(0);
  AT_DISPATCH_SWITCH(out.scalar_type(), "fluxbin_w3_lut_finish",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      launch_finish(partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G, stream);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      launch_finish(partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G, stream);
    })
  );
}

void m1_out(torch::Tensor x, torch::Tensor planes, torch::Tensor scales,
            torch::Tensor perm, torch::Tensor partial, torch::Tensor out,
            int64_t groups_per_split, bool dense_bf16_weight_semantics) {
  TORCH_CHECK(groups_per_split > 0 && groups_per_split <= 255,
              "groups_per_split outside v1 bounds");
  const int64_t G = planes.size(0);
  const int64_t O = planes.size(2);
  const int64_t splits = (G + groups_per_split - 1) / groups_per_split;
  validate_common(x, planes, scales, perm, partial, splits);
  TORCH_CHECK(out.is_cuda() && out.device() == x.device() && out.is_contiguous() &&
              out.sizes() == at::IntArrayRef({1, O}) &&
              out.scalar_type() == x.scalar_type(),
              "out must be contiguous same-device [1,O] with input dtype");
  TORCH_CHECK(!out.is_alias_of(partial), "out must not alias partial");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_w3_lut_m1_out",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      if (dense_bf16_weight_semantics)
        launch_split<scalar_t, true>(
            x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
            scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
            partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G,
            groups_per_split, stream);
      else
        launch_split<scalar_t, false>(
            x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
            scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
            partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G,
            groups_per_split, stream);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      if (dense_bf16_weight_semantics)
        launch_split<scalar_t, true>(
            x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
            scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
            partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G,
            groups_per_split, stream);
      else
        launch_split<scalar_t, false>(
            x.data_ptr<scalar_t>(), planes.data_ptr<uint8_t>(),
            scales.data_ptr<at::BFloat16>(), perm.data_ptr<int16_t>(),
            partial.data_ptr<float>(), out.data_ptr<scalar_t>(), O, G,
            groups_per_split, stream);
    })
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("inline_main", &inline_main);
  module.def("inline_main_bf16_weight", &inline_main_bf16_weight);
  module.def("finish", &finish);
  module.def("m1_out", &m1_out);
}
