#ifndef __softname_kernels_rbf_avx_h
#define __softname_kernels_rbf_avx_h

#include <immintrin.h>

inline __m256d KernelLinear_AVX256(__m256d r, __m256d epsilon)
{
  __m256d u;
  __m256d one = _mm256_set1_pd(1.0);
  u = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  return u;
}

inline __m256d KernelWendlandC2_AVX256(__m256d r, __m256d epsilon)
{
  __m256d u,u2,u4,four_u,inner,rbf,mask;
  __m256d one  = _mm256_set1_pd(1.0);
  __m256d four = _mm256_set1_pd(4.0);
  __m256d five = _mm256_set1_pd(5.0);

  u      = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, r));     // u = 1 - epsilon * r
  mask   = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  u2     = _mm256_mul_pd(u, u);                               // u^2
  u4     = _mm256_mul_pd(u2, u2);                             // u^4
  four_u = _mm256_mul_pd(four, u);                            // 4 * u
  inner  = _mm256_sub_pd(five, four_u);                       // 5 - 4 * u
  rbf    = _mm256_mul_pd(u4, inner);                          // rbf = u^4 * (5 - 4 * u)
  rbf    = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask);  // apply mask to zero out values where u < 0
  return rbf;
}

#endif