/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    kernels_rbf_avx512.h

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#ifndef __LagRBF_kernels_rbf_avx512_h
#define __LagRBF_kernels_rbf_avx512_h

#if defined(__AVX512F__)
#include <immintrin.h>

inline __m512d KernelLinear_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u;
  u = _mm512_mul_pd(epsilon, r); // u = epsilon * r
  return u;
}

inline __m512d KernelWendlandC0_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u,u2;
  __mmask8 mask;
  __m512d one = _mm512_set1_pd(1.0);

  u    = _mm512_sub_pd(one, _mm512_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  mask = _mm512_cmp_pd_mask(u, _mm512_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  u    = _mm512_maskz_mov_pd(mask, u); // apply mask to zero out values where u < 0
  u2   = _mm512_mul_pd(u, u); // u^2
  return u2;
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

inline __m512d KernelBump_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u,neg_inv_u2,denominator,rbf;
  __m512d one = _mm512_set1_pd(1.0);
  __mmask8 mask;
  
  u           = _mm512_sub_pd(one, _mm512_mul_pd(epsilon, r)); // u = 1 - epsilon * r
  mask        = _mm512_cmp_pd_mask(u, _mm512_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
  denominator = _mm512_mul_pd(u, _mm512_sub_pd(_mm512_set1_pd(2.0), u)); // u * (2.0 - u)
  neg_inv_u2  = _mm512_div_pd(_mm512_set1_pd(-1.0), denominator); // -1 / (u * (2.0 - u))
  rbf         = Exp_AVX512(neg_inv_u2); // exp(-1 / (u * (2.0 - u)))
  rbf         = _mm512_maskz_mov_pd(mask, rbf); // apply mask to zero out values where u < 0
  return rbf;
}

inline __m512d KernelGaussian_AVX512(__m512d r, __m512d epsilon)
{
  __m512d u,u2,rbf;
  u   = _mm512_mul_pd(epsilon, r); // u = epsilon * r
  u2  = _mm512_mul_pd(u, u); // u^2
  rbf = Exp_AVX512(_mm512_mul_pd(u2, _mm512_set1_pd(-1.0))); // exp(-u^2)
  return rbf;
}
#endif
#endif
