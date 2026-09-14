/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 * Licensed under the Apache License, Version 2.0. See licenses/Marlin-LICENSE.
 * Modified 2026-09-14: independent hybrid-g128-s8 M=1 adaptation; FP16/BF16,
 * weights in MMA operand A, activation replicated across operand B columns,
 * three-stage copies, register double buffering, persistent tile scheduling.
 * References Marlin revision 1f25790bdd49fba53106164a24666dade68d7c90.
 * Not the original INT4 Marlin kernel. CUDA validation/performance pending.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

// Adapted from Marlin cp_async4_pred/fence/wait. Zero-fill inactive tail rows.
// 8-byte scale loads use .ca; 16-byte packed weights bypass L1 via .cg.
template<int Bytes>
__device__ __forceinline__ void copy_async(void* dst, const void* src, bool valid) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if constexpr (Bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(src), "r"(valid ? 16 : 0) : "memory");
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(smem), "l"(src), "r"(valid ? 8 : 0) : "memory");
}
__device__ __forceinline__ void commit_copy() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int Pending>
__device__ __forceinline__ void wait_copy() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending) : "memory");
}

template<class T> struct alignas(16) Stage {
  // 48-byte row pitch: aligned 16B staging, eight distinct banks for eight
  // MMA row groups reading a fixed packed word (four lanes broadcast per row).
  uint32_t codes[64][12];
  float rows[64][2], sparse_rows[64][2];
  float columns[256], sparse_columns[16];
  T x[128];
  int16_t lookup[128];
  uint16_t sparse[64];
};

template<class T>
__device__ __forceinline__ void fetch_stage(Stage<T>& s, int g, int end, int first_o,
    int O, const T* x, const uint8_t* codes, const float* rows, const float* cols,
    const uint8_t* sparse, const float* sr, const float* sc, const int16_t* lookup) {
  int t = threadIdx.x, row = t / 2, half = t & 1;
  bool group_valid = g < end;
  int safe_g = group_valid ? g : 0;
  bool valid = group_valid && first_o + row < O;
  int safe_o = valid ? first_o + row : 0;
  copy_async<16>(&s.codes[row][half * 4], codes + (safe_g * O + safe_o) * 32 + half * 16, valid);
  if (t < 64) {
    valid = group_valid && first_o + t < O;
    safe_o = valid ? first_o + t : 0;
    copy_async<8>(s.rows[t], rows + (safe_g * O + safe_o) * 2, valid);
    copy_async<8>(s.sparse_rows[t], sr + (safe_g * O + safe_o) * 2, valid);
    copy_async<16>(&s.columns[t * 4], cols + safe_g * 256 + t * 4, group_valid);
    // Two-byte source may be only 2-byte aligned when O is odd. Scalar copy
    // avoids an invalid 4-byte cp.async; all other bulk staging is asynchronous.
    int off = (safe_g * O + safe_o) * 2;
    s.sparse[t] = valid ? uint16_t(sparse[off]) | (uint16_t(sparse[off+1]) << 8) : 0;
  }
  if (t < 16) {
    copy_async<16>(&s.x[t * 8], x + safe_g * 128 + t * 8, group_valid);
    copy_async<16>(&s.lookup[t * 8], lookup + safe_g * 128 + t * 8, group_valid);
  }
  if (t < 4)
    copy_async<16>(&s.sparse_columns[t * 4], sc + safe_g * 16 + t * 4, group_valid);
  commit_copy();
}

__device__ __forceinline__ float pair_value(int bits, float r0, float r1, float c0, float c1) {
  return __fadd_rn(__fmul_rn((bits & 1) ? r0 : -r0, c0),
                  __fmul_rn((bits & 2) ? r1 : -r1, c1));
}
template<class T>
__device__ __forceinline__ T decode_weight(const Stage<T>& s, int row, int k,
                                          float r0, float r1, float s0, float s1, unsigned sp) {
  unsigned word = s.codes[row][k / 16];
  int bits = (word >> (2 * (k & 15))) & 3;
  float w = pair_value(bits,r0,r1,s.columns[2*k],s.columns[2*k+1]);
  int selected = s.lookup[k];
  if (selected >= 0 && selected < 8)
    w = __fadd_rn(w, pair_value((sp >> (2*selected)) & 3,s0,s1,
                               s.sparse_columns[2*selected],s.sparse_columns[2*selected+1]));
  return T(w); // Preserve combined-weight rounding; MMA changes dot reduction order.
}
template<class T>
__device__ __forceinline__ uint32_t pack_pair(T lo, T hi) {
  return uint32_t(lo.x) | (uint32_t(hi.x) << 16);
}
struct Fragments { uint32_t a[4], b[2]; };

template<class T>
__device__ __forceinline__ void load_fragments(Fragments& f, const Stage<T>& s, int step,
    int row, int col, const float (&r)[2][4], const unsigned (&sp)[2]) {
  // NVIDIA m16n8k16 row.col: A registers hold row,row+8 at K halves 0,8.
  #pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    int rr = reg & 1, k = step * 16 + col * 2 + (reg / 2) * 8;
    T lo = decode_weight(s,row+rr*8,k,r[rr][0],r[rr][1],r[rr][2],r[rr][3],sp[rr]);
    T hi = decode_weight(s,row+rr*8,k+1,r[rr][0],r[rr][1],r[rr][2],r[rr][3],sp[rr]);
    f.a[reg] = pack_pair(lo,hi);
  }
  #pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    int k = step * 16 + col * 2 + reg * 8;
    f.b[reg] = pack_pair(s.x[k],s.x[k+1]);
  }
}
// MMA helper adapted from Marlin; BF16 specialization added.
template<class T>
__device__ __forceinline__ void mma(const Fragments& f, float (&c)[4]) {
  if constexpr (std::is_same<T, at::Half>::value)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(f.a[0]),"r"(f.a[1]),"r"(f.a[2]),"r"(f.a[3]),"r"(f.b[0]),"r"(f.b[1]));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(f.a[0]),"r"(f.a[1]),"r"(f.a[2]),"r"(f.a[3]),"r"(f.b[0]),"r"(f.b[1]));
}

template<class T>
__global__ void pipelined_m1(const T* x, const uint8_t* codes, const float* rows,
    const float* cols, const uint8_t* sparse, const float* sr, const float* sc,
    const int16_t* lookup, T* out, float* partial, int O, int G, int gps) {
  __shared__ Stage<T> stages[3];
  const int lane = threadIdx.x & 31;
  const int row = (threadIdx.x / 32) * 16 + lane / 4, col = lane & 3;
  const int tiles = (O + 63) / 64, splits = (G + gps - 1) / gps;
  // Bounded persistent stripes; no spin locks or inter-CTA waiting.
  for (int task = blockIdx.x; task < tiles * splits; task += gridDim.x) {
    int first_o = (task % tiles) * 64, split = task / tiles;
    int start = split * gps, end = min(G,start+gps);
    float accum[4] = {};
    fetch_stage(stages[0],start,end,first_o,O,x,codes,rows,cols,sparse,sr,sc,lookup);
    fetch_stage(stages[1],start+1,end,first_o,O,x,codes,rows,cols,sparse,sr,sc,lookup);
    for (int g = start; g < end; ++g) {
      int i = g-start;
      wait_copy<1>();
      __syncthreads();
      Stage<T>& s = stages[i % 3];
      fetch_stage(stages[(i+2)%3],g+2,end,first_o,O,x,codes,rows,cols,sparse,sr,sc,lookup);
      float r[2][4]; unsigned sp[2];
      #pragma unroll
      for (int rr=0;rr<2;++rr) {
        int ro=row+rr*8;
        r[rr][0]=s.rows[ro][0]; r[rr][1]=s.rows[ro][1];
        r[rr][2]=s.sparse_rows[ro][0]; r[rr][3]=s.sparse_rows[ro][1]; sp[rr]=s.sparse[ro];
      }
      Fragments f[2];
      load_fragments(f[0],s,0,row,col,r,sp);
      #pragma unroll
      for (int step=0;step<8;++step) {
        if (step<7) load_fragments(f[(step+1)&1],s,step+1,row,col,r,sp);
        mma<T>(f[step&1],accum);
      }
      // Every warp must finish reading before a future fetch reuses this slot.
      __syncthreads();
    }
    wait_copy<0>();
    if (col==0) {
      #pragma unroll
      for (int rr=0;rr<2;++rr) {
        int o=first_o+row+rr*8;
        if (o<O) {
          partial[split*O+o]=accum[rr*2];
          if (splits==1) out[o]=T(accum[rr*2]);
        }
      }
    }
    __syncthreads();
  }
}

template<class T>
__global__ void finish_m1(const float* partial, T* out, int O, int splits) {
  int o=blockIdx.x*blockDim.x+threadIdx.x;
  if (o>=O) return;
  float sum=0.f;
  for (int s=0;s<splits;++s) sum+=partial[s*O+o];
  out[o]=T(sum);
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
  for (const auto& t : {x,codes,cols,sc,lookup})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "v3 async input requires 16-byte alignment");
  for (const auto& t : {rows,sr})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 == 0, "v3 scales require 8-byte alignment");
  auto stream = at::cuda::getCurrentCUDAStream();
  int tiles = (O + 63) / 64;
  int blocks = std::min<int64_t>(tiles * splits,
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_m1_v3",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      pipelined_m1<scalar_t><<<blocks,128,0,stream>>>(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), sc.data_ptr<float>(), lookup.data_ptr<int16_t>(), out.data_ptr<scalar_t>(), workspace.data_ptr<float>(), O,G,gps);
      if (splits > 1) finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>(), out.data_ptr<scalar_t>(),O,splits);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      pipelined_m1<scalar_t><<<blocks,128,0,stream>>>(x.data_ptr<scalar_t>(), codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), cols.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), sc.data_ptr<float>(), lookup.data_ptr<int16_t>(), out.data_ptr<scalar_t>(), workspace.data_ptr<float>(), O,G,gps);
      if (splits > 1) finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>(), out.data_ptr<scalar_t>(),O,splits);
    })
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("m1_out", &m1_out); }
