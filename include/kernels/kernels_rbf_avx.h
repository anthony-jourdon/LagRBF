#ifndef __LagRBF_kernels_rbf_avx_h
#define __LagRBF_kernels_rbf_avx_h

#if defined(__AVX__)
#include <immintrin.h>
#include "kernels/exp_avx.h"

inline __m256d KernelLinear_AVX256(__m256d r, __m256d epsilon)
{
  __m256d u;
  u = _mm256_mul_pd(epsilon, r); // u = epsilon * r
  return u;
}

inline __m256d KernelWendlandC0_AVX256(__m256d r, __m256d epsilon)
{
  __m256d u,u2;
  __m256d mask;
  __m256d one = _mm256_set1_pd(1.0);

  u    = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  mask = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  u    = _mm256_blendv_pd(_mm256_setzero_pd(), u, mask); // apply mask to zero out values where u < 0
  u2   = _mm256_mul_pd(u, u); // u^2
  return u2;
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

inline __m256d KernelBump_AVX256(__m256d r, __m256d epsilon)
{
  __m256d u,neg_inv_u2,denominator,rbf,mask;
  __m256d one = _mm256_set1_pd(1.0);
  
  u           = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  mask        = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  denominator = _mm256_mul_pd(u, _mm256_sub_pd(_mm256_set1_pd(2.0), u)); // u * (2.0 - u)
  neg_inv_u2  = _mm256_div_pd(_mm256_set1_pd(-1.0), denominator); // -1 / (u * (2.0 - u))
  rbf         = Exp_AVX256(neg_inv_u2); // exp(-1 / (u * (2.0 - u)))
  rbf         = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask); // apply mask to zero out values where u < 0
  return rbf;
}
#endif
#endif