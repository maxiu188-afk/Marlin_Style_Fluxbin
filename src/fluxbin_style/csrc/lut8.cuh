// Shared host/device primitives for the FP32 eight-sign activation LUT.
#pragma once
#ifdef __CUDACC__
#define FLUXBIN_INLINE __host__ __device__ __forceinline__
#else
#define FLUXBIN_INLINE inline
#endif
namespace fluxbin_lut {
// Extract one base's eight sign bits from sixteen interleaved bits.
FLUXBIN_INLINE unsigned pattern(unsigned word) {
  word &= 0x5555u;
  word = (word | (word >> 1)) & 0x3333u;
  word = (word | (word >> 2)) & 0x0f0fu;
  return (word | (word >> 4)) & 0xffu;
}
// Each lane owns eight table entries. Recurrence reads only that lane's
// previously written entries, so no inter-lane synchronization is needed here.
FLUXBIN_INLINE void build_lane(float* table, const float* values, unsigned lane) {
  float sum = 0.f;
  for (int bit = 0; bit < 8; ++bit)
    sum += ((lane >> bit) & 1u) ? values[bit] : -values[bit];
  table[lane] = sum;
  for (unsigned bit = 5; bit < 8; ++bit) {
    float increment = values[bit] + values[bit];
    for (unsigned p = 1u << bit; p < (1u << (bit + 1)); p += 32)
      table[p + lane] = table[p + lane - (1u << bit)] + increment;
  }
}
} // namespace fluxbin_lut
#undef FLUXBIN_INLINE
