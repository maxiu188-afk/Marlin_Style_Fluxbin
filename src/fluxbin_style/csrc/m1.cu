// Original FluxBin M=1 research kernel. Marlin-inspired tiling/packed loads;
// this is SIMT, not Marlin INT4 MMA. No code or INT4 re-quantization is imported.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float pair_value(int code, float r0, float r1,
                                           float c0, float c1) {
  // Preserve separate FP32 multiplications and addition used by the oracle.
  float a = __fmul_rn((code & 1) ? r0 : -r0, c0);
  float b = __fmul_rn((code & 2) ? r1 : -r1, c1);
  return __fadd_rn(a, b);
}

template<class T>
__global__ void partial_m1(const T* x, const uint8_t* codes, const float* rows,
                          const float* cols, const uint8_t* sparse,
                          const float* sr, const float* sc, const int16_t* lookup,
                          float* partial, int O, int G, int gps) {
  // Four output rows, one warp per row; adjacent lanes read adjacent packed bytes.
  const int lane = threadIdx.x & 31, o = blockIdx.x * 4 + threadIdx.x / 32;
  __shared__ float sx[128], c0[128], c1[128], cs[16];
  __shared__ int16_t map[128];
  float accum = 0.f;
  int end = min(G, (int(blockIdx.y) + 1) * gps);
  for (int g = blockIdx.y * gps; g < end; ++g) {
    int t = threadIdx.x;
    sx[t] = float(x[g * 128 + t]);
    c0[t] = cols[g * 256 + t * 2];
    c1[t] = cols[g * 256 + t * 2 + 1];
    map[t] = lookup[g * 128 + t];
    if (t < 16) cs[t] = sc[g * 16 + t];
    __syncthreads();
    if (o < O) {
      int off = (g * O + o) * 2;
      float r0 = rows[off], r1 = rows[off + 1];
      float s0 = sr[off], s1 = sr[off + 1];
      unsigned q = codes[(g * O + o) * 32 + lane];
      unsigned sp = unsigned(sparse[off]) | (unsigned(sparse[off + 1]) << 8);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int k = lane * 4 + j, selected = map[k];
        float w = pair_value((q >> (2*j)) & 3, r0, r1, c0[k], c1[k]);
        if (selected >= 0 && selected < 8) {
          float delta = pair_value((sp >> (2*selected)) & 3, s0, s1,
                                   cs[2*selected], cs[2*selected+1]);
          w = __fadd_rn(w, delta);
        }
        // Critical: round the COMBINED weight to activation dtype before dot.
        // Factoring row scales outside dot would change accepted BF16 semantics.
        float rounded = float(T(w));
        accum = __fmaf_rn(sx[k], rounded, accum);
      }
    }
    __syncthreads();
  }
  for (int delta = 16; delta; delta >>= 1)
    accum += __shfl_down_sync(0xffffffff, accum, delta);
  if (o < O && lane == 0) partial[blockIdx.y * O + o] = accum;
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
  dim3 grid((O+3)/4, splits);
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_m1",
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
