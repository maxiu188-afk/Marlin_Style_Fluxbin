// Shared host/device primitive for an unsigned eight-value subset-sum LUT.
#pragma once
#ifdef __CUDACC__
#define FLUXBIN_W3_INLINE __host__ __device__ __forceinline__
#else
#define FLUXBIN_W3_INLINE inline
#endif

namespace fluxbin_w3_lut {

// Each lane owns eight entries. The first value covers the low five bits of
// the pattern; the recurrence adds bits 5..7 without inter-lane communication.
FLUXBIN_W3_INLINE void build_lane_unsigned(float* table, const float* values,
                                           unsigned lane) {
  float sum = 0.f;
  #pragma unroll
  for (int bit = 0; bit < 8; ++bit)
    sum += ((lane >> bit) & 1u) ? values[bit] : 0.f;
  table[lane] = sum;
  #pragma unroll
  for (unsigned bit = 5; bit < 8; ++bit) {
    const float increment = values[bit];
    #pragma unroll
    for (unsigned pattern = 1u << bit; pattern < (1u << (bit + 1)); pattern += 32)
      table[pattern + lane] = table[pattern + lane - (1u << bit)] + increment;
  }
}

}  // namespace fluxbin_w3_lut

#undef FLUXBIN_W3_INLINE
