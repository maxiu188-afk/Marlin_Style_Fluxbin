// Local v2 candidate: bank-swizzled shared memory and 4/8/16 output rows/CTA.
// Preserves v1 artifact/rounding/split reduction; CUDA validation pending.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef ROWS_PER_WARP
#define ROWS_PER_WARP 4
#endif
static_assert(ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4);

__device__ __forceinline__ float pair_value(int code, float r0, float r1,
                                           float c0, float c1) {
  // Preserve separate FP32 multiplications and addition used by the oracle.
  float a = __fmul_rn((code & 1) ? r0 : -r0, c0);
  float b = __fmul_rn((code & 2) ? r1 : -r1, c1);
  return __fadd_rn(a, b);
}

// A bijection of 128 scalar positions. Both contiguous staging writes and
// k=4*lane+j reads hit distinct 32-bit shared-memory banks within each warp.
__device__ __forceinline__ int shared_index(int k) {
  return (k & 3) * 32 + ((k >> 2) ^ ((k & 3) * 8));
}

template<class T>
__global__ void partial_m1(const T* x, const uint8_t* codes, const float* rows,
                          const float* cols, const uint8_t* sparse,
                          const float* sr, const float* sc, const int16_t* lookup,
                          float* partial, int O, int G, int gps) {
  constexpr int ROWS = ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int first_o = blockIdx.x * (4 * ROWS) + (threadIdx.x / 32) * ROWS;
  __shared__ float sx[128], c0[128], c1[128], cs[16];
  __shared__ int map[128]; // 32-bit to match the bank-address mapping.
  float accum[ROWS] = {};
  int end = min(G, (int(blockIdx.y) + 1) * gps);
  for (int g = blockIdx.y * gps; g < end; ++g) {
    int t = threadIdx.x, sw = shared_index(t);
    sx[sw] = float(x[g * 128 + t]);
    c0[sw] = cols[g * 256 + t * 2];
    c1[sw] = cols[g * 256 + t * 2 + 1];
    map[sw] = lookup[g * 128 + t];
    if (t < 16) cs[t] = sc[g * 16 + t];
    __syncthreads();
    float r0[ROWS], r1[ROWS], s0[ROWS], s1[ROWS];
    unsigned q[ROWS], sp[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int o = first_o + r;
      r0[r] = r1[r] = s0[r] = s1[r] = 0.f;
      q[r] = sp[r] = 0;
      if (o < O) {
        int off = (g * O + o) * 2;
        r0[r] = rows[off]; r1[r] = rows[off + 1];
        s0[r] = sr[off]; s1[r] = sr[off + 1];
        q[r] = codes[(g * O + o) * 32 + lane];
        sp[r] = unsigned(sparse[off]) | (unsigned(sparse[off + 1]) << 8);
      }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int k = lane * 4 + j, index = shared_index(k), selected = map[index];
      float activation = sx[index], col0 = c0[index], col1 = c1[index];
      bool has_sparse = selected >= 0 && selected < 8;
      float sparse0 = 0.f, sparse1 = 0.f;
      if (has_sparse) { sparse0 = cs[2*selected]; sparse1 = cs[2*selected+1]; }
      // Reuse activation/columns/index across the selected rows in registers.
      #pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        if (first_o + r < O) {
          float w = pair_value((q[r] >> (2*j)) & 3, r0[r], r1[r], col0, col1);
          if (has_sparse) {
            float delta = pair_value((sp[r] >> (2*selected)) & 3,
                                     s0[r], s1[r], sparse0, sparse1);
            w = __fadd_rn(w, delta);
          }
          // Same per-row group/j order and combined-weight rounding as v1.
          accum[r] = __fmaf_rn(activation, float(T(w)), accum[r]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    for (int delta = 16; delta; delta >>= 1)
      accum[r] += __shfl_down_sync(0xffffffff, accum[r], delta);
    if (first_o + r < O && lane == 0)
      partial[blockIdx.y * O + first_o + r] = accum[r];
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

void m1_out(torch::Tensor x, torch::Tensor codes, torch::Tensor rows,
            torch::Tensor cols, torch::Tensor sparse, torch::Tensor sr,
            torch::Tensor sc, torch::Tensor lookup, torch::Tensor out,
            torch::Tensor workspace, int64_t gps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) == 1, "CUDA [1,K] required");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "SM80 or newer required");
  for (const auto& t : {x, codes, rows, cols, sparse, sr, sc, lookup, out, workspace}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous(),
                "all tensors must be contiguous on input CUDA device");
  }
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "BF16/FP16 only");
  TORCH_CHECK(codes.scalar_type() == at::kByte && sparse.scalar_type() == at::kByte,
              "uint8 codes required");
  TORCH_CHECK(lookup.scalar_type() == at::kShort, "int16 lookup required");
  for (const auto& t : {rows, cols, sr, sc, workspace})
    TORCH_CHECK(t.scalar_type() == at::kFloat, "FP32 scales/workspace required");
  TORCH_CHECK(codes.dim() == 3 && codes.size(2) == 32, "codes must be [G,O,32]");
  int64_t G = codes.size(0), O = codes.size(1);
  TORCH_CHECK(G > 0 && O > 0 && G <= 1024 && O <= 65536 && gps > 0 && gps <= 1024,
              "dimensions/split outside v1 bounds");
  int64_t splits = (G + gps - 1) / gps;
  TORCH_CHECK(x.size(1) == G*128, "K mismatch");
  TORCH_CHECK(rows.sizes() == at::IntArrayRef({G,O,2}) && sr.sizes() == rows.sizes(), "row shape");
  TORCH_CHECK(cols.sizes() == at::IntArrayRef({G,128,2}), "column shape");
  TORCH_CHECK(sparse.sizes() == at::IntArrayRef({G,O,2}), "sparse code shape");
  TORCH_CHECK(sc.sizes() == at::IntArrayRef({G,8,2}), "sparse column shape");
  TORCH_CHECK(lookup.sizes() == at::IntArrayRef({G,128}), "lookup shape");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({1,O}) && out.scalar_type() == x.scalar_type(), "output mismatch");
  TORCH_CHECK(workspace.sizes() == at::IntArrayRef({splits,O}), "workspace shape mismatch");
  for (const auto& t : {x,codes,rows,cols,sparse,sr,sc,lookup}) {
    TORCH_CHECK(!out.is_alias_of(t) && !workspace.is_alias_of(t), "input/output alias forbidden");
  }
  TORCH_CHECK(!out.is_alias_of(workspace), "output/workspace alias forbidden");
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int TILE_O = 4 * ROWS_PER_WARP;
  dim3 grid((O+TILE_O-1)/TILE_O, splits);
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_m1_v2",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      partial_m1<scalar_t><<<grid,128,0,stream>>>(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), sc.data_ptr<float>(), lookup.data_ptr<int16_t>(), workspace.data_ptr<float>(), O,G,gps);
      finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>(), out.data_ptr<scalar_t>(),O,splits);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      partial_m1<scalar_t><<<grid,128,0,stream>>>(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), sc.data_ptr<float>(), lookup.data_ptr<int16_t>(), workspace.data_ptr<float>(), O,G,gps);
      finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>(), out.data_ptr<scalar_t>(),O,splits);
    })
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("m1_out", &m1_out); }
