// FluxBin v4_late: factored FP32 sign dot, no per-weight BF16 reconstruction.
// Late warp reduction after each K split; v1-v3 remain unchanged. CUDA validation pending.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Transform once per group for ALL output rows. Permute K for conflict-free
// shared accesses when each lane consumes four consecutive packed signs.
template<class T>
__global__ void prepare_x(const T* x, const float* cols, const float* sc,
                          const int16_t* indices, float* z) {
  int g=blockIdx.x, t=threadIdx.x, p=(t%4)*32+t/4;
  float a=float(x[g*128+t]);
  z[g*272+p]=__fmul_rn(a,cols[g*256+2*t]);
  z[g*272+128+p]=__fmul_rn(a,cols[g*256+2*t+1]);
  if(t<8) {
    int index=indices[g*8+t];
    float b=(index>=0 && index<128) ? float(x[g*128+index]) : __int_as_float(0x7fffffff);
    z[g*272+256+t]=__fmul_rn(b,sc[g*16+2*t]);
    z[g*272+264+t]=__fmul_rn(b,sc[g*16+2*t+1]);
  }
}
__device__ __forceinline__ float signed_value(float v, unsigned positive) {
  return __uint_as_float(__float_as_uint(v) ^ (positive ? 0u : 0x80000000u));
}
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for(int d=16;d;d>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));
  return v;
}
__global__ void factored_m1(const uint8_t* codes, const float* rows,
                            const uint8_t* sparse, const float* sr,
                            const float* z, float* partial, int O,int G,int gps) {
  __shared__ float sz[272];
  int t=threadIdx.x, lane=t%32, first=blockIdx.x*16+(t/32)*4;
  float result[4]={};
  int end=min(G,(int(blockIdx.y)+1)*gps);
  for(int g=blockIdx.y*gps;g<end;++g) {
    sz[t]=z[g*272+t]; sz[128+t]=z[g*272+128+t];
    if(t<16) sz[256+t]=z[g*272+256+t];
    __syncthreads();
    // One packed byte per lane per row, four weights per byte. Activation
    // transforms are reused by the four output rows of each warp.
    float a[4],b[4];
    #pragma unroll
    for(int j=0;j<4;++j) { a[j]=sz[j*32+lane]; b[j]=sz[128+j*32+lane]; }
    #pragma unroll
    for(int r=0;r<4;++r) {
      int o=first+r;
      if(o<O) { // uniform within each warp; all lanes participate in reductions
        unsigned q=codes[(g*O+o)*32+lane];
        float d0=0.f,d1=0.f;
        #pragma unroll
        for(int j=0;j<4;++j) {
          d0=__fadd_rn(d0,signed_value(a[j],q&(1u<<(2*j))));
          d1=__fadd_rn(d1,signed_value(b[j],q&(2u<<(2*j))));
        }
        float s0=0.f,s1=0.f;
        if(lane<8) {
          int off=(g*O+o)*2;
          unsigned sp=unsigned(sparse[off])|(unsigned(sparse[off+1])<<8);
          s0=signed_value(sz[256+lane],sp&(1u<<(2*lane)));
          s1=signed_value(sz[264+lane],sp&(2u<<(2*lane)));
        }
        { // Every lane applies row scales before the final warp reduction.
          int off=(g*O+o)*2;
          result[r]=__fmaf_rn(rows[off],d0,result[r]);
          result[r]=__fmaf_rn(rows[off+1],d1,result[r]);
          result[r]=__fmaf_rn(sr[off],s0,result[r]);
          result[r]=__fmaf_rn(sr[off+1],s1,result[r]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<4;++r) {
    float sum=warp_sum(result[r]);
    if(lane==0 && first+r<O) partial[blockIdx.y*O+first+r]=sum;
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
  TORCH_CHECK(indices.scalar_type() == at::kShort, "int16 indices required");
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
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({G,8}), "v4 indices shape");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({1,O}) && out.scalar_type() == x.scalar_type(), "output mismatch");
  TORCH_CHECK(workspace.sizes() == at::IntArrayRef({G*272+splits*O}), "workspace shape mismatch");
  for (const auto& t : {x,codes,rows,cols,sparse,sr,sc,indices}) {
    TORCH_CHECK(!out.is_alias_of(t) && !workspace.is_alias_of(t), "input/output alias forbidden");
  }
  TORCH_CHECK(!out.is_alias_of(workspace), "output/workspace alias forbidden");
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((O+15)/16, splits);
  AT_DISPATCH_SWITCH(x.scalar_type(), "fluxbin_m1_v4_late",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      prepare_x<scalar_t><<<G,128,0,stream>>>(x.data_ptr<scalar_t>(), cols.data_ptr<float>(), sc.data_ptr<float>(), indices.data_ptr<int16_t>(), workspace.data_ptr<float>());
      factored_m1<<<grid,128,0,stream>>>(codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), workspace.data_ptr<float>(), workspace.data_ptr<float>()+G*272, O,G,gps);
      finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>()+G*272, out.data_ptr<scalar_t>(),O,splits);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      prepare_x<scalar_t><<<G,128,0,stream>>>(x.data_ptr<scalar_t>(), cols.data_ptr<float>(), sc.data_ptr<float>(), indices.data_ptr<int16_t>(), workspace.data_ptr<float>());
      factored_m1<<<grid,128,0,stream>>>(codes.data_ptr<uint8_t>(), rows.data_ptr<float>(), sparse.data_ptr<uint8_t>(), sr.data_ptr<float>(), workspace.data_ptr<float>(), workspace.data_ptr<float>()+G*272, O,G,gps);
      finish_m1<scalar_t><<<(O+255)/256,256,0,stream>>>(workspace.data_ptr<float>()+G*272, out.data_ptr<scalar_t>(),O,splits);
    })
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("m1_out", &m1_out); }
