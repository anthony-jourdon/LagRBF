#ifndef __softname_kernels_rbf_avx512_h
#define __softname_kernels_rbf_avx512_h

#if defined(__AVX512F__)
#include <immintrin.h>

inline __m512d KernelLinear_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u;
  __m512d one = _mm512_set1_pd(1.0);
  u = _mm512_sub_pd(one, _mm512_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  return u;
}

inline __m512d KernelWendlandC2_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u,u2,u4,four_u,inner,rbf;
  __mmask8 mask;
  __m512d one  = _mm512_set1_pd(1.0);
  __m512d four = _mm512_set1_pd(4.0);
  __m512d five = _mm512_set1_pd(5.0);

  u      = _mm512_sub_pd(one, _mm512_mul_pd(epsilon, r));     // u = 1 - epsilon * r
  mask   = _mm512_cmp_pd_mask(u, _mm512_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  u2     = _mm512_mul_pd(u, u);                               // u^2
  u4     = _mm512_mul_pd(u2, u2);                             // u^4
  four_u = _mm512_mul_pd(four, u);                            // 4 * u
  inner  = _mm512_sub_pd(five, four_u);                       // 5 - 4 * u
  rbf    = _mm512_mul_pd(u4, inner);                          // rbf = u^4 * (5 - 4 * u)
  rbf    = _mm512_maskz_mov_pd(mask, rbf);                    // apply mask to zero out values where u < 0
  return rbf;
}
#endif
#endif
