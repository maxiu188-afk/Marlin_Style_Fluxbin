// FluxBin v5 LUT-A16: FP32 activation tables, no activation quantization.
// QBB-style eight-sign lookup and row ownership, adapted to two distinct
// column-scale bases plus sparse refinement. CUDA validation pending.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include "lut8.cuh"

#ifndef PLANAR_CODES
#define PLANAR_CODES 0
#endif
#ifndef LUT_ROWS
#define LUT_ROWS 1024
#endif
constexpr int kThreads = 256;
constexpr int kRows = LUT_ROWS;
constexpr int kRowsPerThread = kRows / kThreads;
static_assert(kRows == 256 || kRows == 512 || kRows == 1024, "unsupported row tile");
constexpr int kTables = 34; // 16 global tables/base, one sparse table/base

template<class T, bool SingleSplit>
__global__ void lut_m1(const T* x, const uint8_t* codes, const float* rows,
                       const float* cols, const uint8_t* sparse,
                       const float* sr, const float* sc, const int16_t* indices,
                       float* partial, T* out, int O, int G, int gps) {
  __shared__ float lut[kTables * 256];
  int t = threadIdx.x, lane = t % 32, warp = t / 32;
  int first = blockIdx.x * kRows + t;
  float result[kRowsPerThread] = {};
  int end = min(G, (int(blockIdx.y) + 1) * gps);
  for (int g = blockIdx.y * gps; g < end; ++g) {
    // Build tables directly from x*column in this CTA. The output row tile
    // amortize construction; no transformed-activation global workspace.
    for (int bank = warp; bank < kTables; bank += kThreads / 32) {
      float values[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (bank < 32) {
          int basis = bank / 16, k = (bank % 16) * 8 + j;
          values[j] = __fmul_rn(float(x[g*128+k]), cols[g*256+2*k+basis]);
        } else {
          int index = indices[g*8+j];
          float a = (index >= 0 && index < 128) ? float(x[g*128+index])
                                                  : __int_as_float(0x7fffffff);
          values[j] = __fmul_rn(a, sc[g*16+2*j+bank-32]);
        }
      }
      fluxbin_lut::build_lane(lut + bank*256, values, lane);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      int o = first + r*kThreads;
      if (o < O) {
        // Each row occupies exactly 32 aligned bytes. Vector loads avoid
        // issuing sixteen separate strided two-byte global loads per row.
        const uint4* ptr = reinterpret_cast<const uint4*>(codes + (g*O+o)*32);
        uint4 lo = ptr[0], hi = ptr[1];
        unsigned words[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        float d0 = 0.f, d1 = 0.f;
        #pragma unroll
        for (int chunk = 0; chunk < 16; ++chunk) {
#if PLANAR_CODES
          unsigned p0 = (words[chunk/4] >> (8*(chunk%4))) & 0xffu;
          unsigned p1 = (words[4+chunk/4] >> (8*(chunk%4))) & 0xffu;
#else
          unsigned word = words[chunk/2] >> (16*(chunk%2));
          unsigned p0 = fluxbin_lut::pattern(word);
          unsigned p1 = fluxbin_lut::pattern(word >> 1);
#endif
          d0 += lut[chunk*256 + p0];
          d1 += lut[(16+chunk)*256 + p1];
        }
        unsigned sp = reinterpret_cast<const uint16_t*>(sparse)[g*O+o];
#if PLANAR_CODES
        float s0 = lut[32*256 + (sp & 0xffu)];
        float s1 = lut[33*256 + (sp >> 8)];
#else
        float s0 = lut[32*256 + fluxbin_lut::pattern(sp)];
        float s1 = lut[33*256 + fluxbin_lut::pattern(sp >> 1)];
#endif
        int off = (g*O+o)*2;
        result[r] = __fmaf_rn(rows[off], d0, result[r]);
        result[r] = __fmaf_rn(rows[off+1], d1, result[r]);
        result[r] = __fmaf_rn(sr[off], s0, result[r]);
        result[r] = __fmaf_rn(sr[off+1], s1, result[r]);
      }
    }
    __syncthreads(); // all consumers finish before the next group overwrites LUT
  }
  #pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    int o = first + r*kThreads;
    if (o < O) {
      partial[blockIdx.y*O+o] = result[r];
      if constexpr (SingleSplit) out[o] = T(result[r]);
    }
  }
}
template<class T>
__global__ void finish_m1(const float* partial, T* out, int O, int splits) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float value = 0.f;
  for (int s = 0; s < splits; ++s) value += partial[s * O + o];
  out[o] = T(value);
}

template<class T>
void launch(const T* x, const uint8_t* codes, const float* rows, const float* cols,
            const uint8_t* sparse, const float* sr, const float* sc,
            const int16_t* indices, float* partial, T* out,
            int O, int G, int gps, cudaStream_t stream) {
  int splits = (G+gps-1)/gps;
  dim3 grid((O+kRows-1)/kRows, splits);
  if (splits == 1) {
    lut_m1<T,true><<<grid,kThreads,0,stream>>>(x,codes,rows,cols,sparse,sr,sc,indices,partial,out,O,G,gps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lut_m1<T,false><<<grid,kThreads,0,stream>>>(x,codes,rows,cols,sparse,sr,sc,indices,partial,out,O,G,gps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finish_m1<T><<<(O+255)/256,256,0,stream>>>(partial,out,O,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void m1_out(torch::Tensor x, torch::Tensor codes, torch::Tensor rows,
            torch::Tensor cols, torch::Tensor sparse, torch::Tensor sr,
            torch::Tensor sc, torch::Tensor indices, torch::Tensor out,
            torch::Tensor workspace, int64_t gps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) == 1, "CUDA [1,K] required");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "SM80 or newer required");
  for (const auto& t : {x, codes, rows, cols, sparse, sr, sc, indices, out, workspace}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous(),
                "all tensors must be contiguous on input CUDA device");
  }
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "BF16/FP16 only");
  TORCH_CHECK(codes.scalar_type() == at::kByte && sparse.scalar_type() == at::kByte,
              "uint8 codes required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(sparse.data_ptr()) % 2 == 0,
              "v5 requires aligned code buffers; use convert_artifact");
  TORCH_CHECK(indices.scalar_type() == at::kShort, "int16 indices required");
  for (const auto& t : {rows, cols, sr, sc, workspace})
    TORCH_CHECK(t.scalar_type() == at::kFloat, "FP32 scales/workspace required");
#if PLANAR_CODES
  TORCH_CHECK(codes.dim() == 4 && codes.size(2) == 2 && codes.size(3) == 16,
              "planar codes must be [G,O,2,16]");
#else
  TORCH_CHECK(codes.dim() == 3 && codes.size(2) == 32, "codes must be [G,O,32]");
#endif
  int64_t G = codes.size(0), O = codes.size(1);
  TORCH_CHECK(G > 0 && O > 0 && G <= 1024 && O <= 65536 && gps > 0 && gps <= 1024,
              "dimensions/split outside v1 bounds");
  int64_t splits = (G + gps - 1) / gps;
  TORCH_CHECK(x.size(1) == G*128, "K mismatch");
  TORCH_CHECK(rows.sizes() == at::IntArrayRef({G,O,2}) && sr.sizes() == rows.sizes(), "row shape");
  TORCH_CHECK(cols.sizes() == at::IntArrayRef({G,128,2}), "column shape");
#if PLANAR_CODES
  TORCH_CHECK(sparse.sizes() == at::IntArrayRef({G,O,2,1}), "planar sparse code shape");
#else
  TORCH_CHECK(sparse.sizes() == at::IntArrayRef({G,O,2}), "sparse code shape");
#endif
  TORCH_CHECK(sc.sizes() == at::IntArrayRef({G,8,2}), "sparse column shape");
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({G,8}), "v5 indices shape");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({1,O}) && out.scalar_type() == x.scalar_type(), "output mismatch");
  TORCH_CHECK(workspace.sizes() == at::IntArrayRef({splits,O}), "workspace shape mismatch");
  for (const auto& t : {x,codes,rows,cols,sparse,sr,sc,indices}) {
    TORCH_CHECK(!out.is_alias_of(t) && !workspace.is_alias_of(t), "input/output alias forbidden");
  }
  TORCH_CHECK(!out.is_alias_of(workspace), "output/workspace alias forbidden");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_m1_v5",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      launch(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(),
             cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(),
             sc.data_ptr<float>(), indices.data_ptr<int16_t>(), workspace.data_ptr<float>(),
             out.data_ptr<scalar_t>(), O,G,gps,stream);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      launch(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(),
             cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(),
             sc.data_ptr<float>(), indices.data_ptr<int16_t>(), workspace.data_ptr<float>(),
             out.data_ptr<scalar_t>(), O,G,gps,stream);
    })
  );
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("m1_out", &m1_out); }
