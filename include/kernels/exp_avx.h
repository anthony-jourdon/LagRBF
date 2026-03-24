#ifndef __softname_exp_avx_h
#define __softname_exp_avx_h

#include <immintrin.h>

#if defined(__AVX__)
inline __m256d Exp_AVX256(__m256d x)
{
	const __m256d one    = _mm256_set1_pd(1.0);
	const __m256d half   = _mm256_set1_pd(0.5);
	const __m256d log2ef = _mm256_set1_pd(1.44269504088896340736);
	const __m256d exp_c1 = _mm256_set1_pd(6.93145751953125000000e-1);
	const __m256d exp_c2 = _mm256_set1_pd(1.42860682030941723212e-6);
	const __m256d exp_hi = _mm256_set1_pd(7.09782712893383973096e2);
	const __m256d exp_lo = _mm256_set1_pd(-7.08396418532264106224e2);
	const __m256d p0     = _mm256_set1_pd(1.26177193074810590878e-4);
	const __m256d p1     = _mm256_set1_pd(3.02994407707441961300e-2);
	const __m256d p2     = _mm256_set1_pd(9.99999999999999999910e-1);
	const __m256d q0     = _mm256_set1_pd(3.00198505138664455042e-6);
	const __m256d q1     = _mm256_set1_pd(2.52448340349684104192e-3);
	const __m256d q2     = _mm256_set1_pd(2.27265548208155028766e-1);
	const __m256d q3     = _mm256_set1_pd(2.00000000000000000009e0);
	const __m256d inf    = _mm256_set1_pd(__builtin_inf());
	const __m256d zero   = _mm256_setzero_pd();

	__m256d x_clamped, fx, tmp, mask, z, px, qx, y;
	__m128i emm0, emm0_biased, emm0_lo64, emm0_hi64;
	__m256i pow2n_bits;
	__m256d overflow_mask, underflow_mask, nan_mask;

	overflow_mask = _mm256_cmp_pd(x, exp_hi, _CMP_GT_OQ);
	underflow_mask = _mm256_cmp_pd(x, exp_lo, _CMP_LT_OQ);
	nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

	x_clamped = _mm256_min_pd(x, exp_hi);
	x_clamped = _mm256_max_pd(x_clamped, exp_lo);

	fx = _mm256_add_pd(_mm256_mul_pd(x_clamped, log2ef), half);
	emm0 = _mm256_cvttpd_epi32(fx);
	tmp = _mm256_cvtepi32_pd(emm0);

	mask = _mm256_cmp_pd(tmp, fx, _CMP_GT_OQ);
	mask = _mm256_and_pd(mask, one);
	fx = _mm256_sub_pd(tmp, mask);

	x_clamped = _mm256_sub_pd(x_clamped, _mm256_mul_pd(fx, exp_c1));
	x_clamped = _mm256_sub_pd(x_clamped, _mm256_mul_pd(fx, exp_c2));

	z = _mm256_mul_pd(x_clamped, x_clamped);

	px = _mm256_add_pd(_mm256_mul_pd(p0, z), p1);
	px = _mm256_add_pd(_mm256_mul_pd(px, z), p2);
	px = _mm256_mul_pd(px, x_clamped);

	qx = _mm256_add_pd(_mm256_mul_pd(q0, z), q1);
	qx = _mm256_add_pd(_mm256_mul_pd(qx, z), q2);
	qx = _mm256_add_pd(_mm256_mul_pd(qx, z), q3);

	y = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
	y = _mm256_add_pd(one, _mm256_add_pd(y, y));

	emm0_biased = _mm_add_epi32(emm0, _mm_set1_epi32(1023));
	emm0_lo64 = _mm_unpacklo_epi32(emm0_biased, _mm_setzero_si128());
	emm0_hi64 = _mm_unpackhi_epi32(emm0_biased, _mm_setzero_si128());
	emm0_lo64 = _mm_slli_epi64(emm0_lo64, 52);
	emm0_hi64 = _mm_slli_epi64(emm0_hi64, 52);
	pow2n_bits = _mm256_castsi128_si256(emm0_lo64);
	pow2n_bits = _mm256_insertf128_si256(pow2n_bits, emm0_hi64, 1);
	y = _mm256_mul_pd(y, _mm256_castsi256_pd(pow2n_bits));

	y = _mm256_blendv_pd(y, zero, underflow_mask);
	y = _mm256_blendv_pd(y, inf, overflow_mask);
	y = _mm256_blendv_pd(y, x, nan_mask);
	return y;
}
#endif

#if defined(__AVX512F__)
inline __m512d Exp_AVX512(__m512d x)
{
	const __m512d one    = _mm512_set1_pd(1.0);
	const __m512d half   = _mm512_set1_pd(0.5);
	const __m512d log2ef = _mm512_set1_pd(1.44269504088896340736);
	const __m512d exp_c1 = _mm512_set1_pd(6.93145751953125000000e-1);
	const __m512d exp_c2 = _mm512_set1_pd(1.42860682030941723212e-6);
	const __m512d exp_hi = _mm512_set1_pd(7.09782712893383973096e2);
	const __m512d exp_lo = _mm512_set1_pd(-7.08396418532264106224e2);
	const __m512d p0     = _mm512_set1_pd(1.26177193074810590878e-4);
	const __m512d p1     = _mm512_set1_pd(3.02994407707441961300e-2);
	const __m512d p2     = _mm512_set1_pd(9.99999999999999999910e-1);
	const __m512d q0     = _mm512_set1_pd(3.00198505138664455042e-6);
	const __m512d q1     = _mm512_set1_pd(2.52448340349684104192e-3);
	const __m512d q2     = _mm512_set1_pd(2.27265548208155028766e-1);
	const __m512d q3     = _mm512_set1_pd(2.00000000000000000009e0);
	const __m512d inf    = _mm512_set1_pd(__builtin_inf());
	const __m512d zero   = _mm512_setzero_pd();

	__m512d x_clamped, n, z, px, qx, y;
	__mmask8 overflow_mask, underflow_mask, nan_mask;

	overflow_mask = _mm512_cmp_pd_mask(x, exp_hi, _CMP_GT_OQ);
	underflow_mask = _mm512_cmp_pd_mask(x, exp_lo, _CMP_LT_OQ);
	nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);

	x_clamped = _mm512_min_pd(x, exp_hi);
	x_clamped = _mm512_max_pd(x_clamped, exp_lo);

	n = _mm512_add_pd(_mm512_mul_pd(x_clamped, log2ef), half);
	n = _mm512_floor_pd(n);

	x_clamped = _mm512_sub_pd(x_clamped, _mm512_mul_pd(n, exp_c1));
	x_clamped = _mm512_sub_pd(x_clamped, _mm512_mul_pd(n, exp_c2));

	z = _mm512_mul_pd(x_clamped, x_clamped);

	px = _mm512_add_pd(_mm512_mul_pd(p0, z), p1);
	px = _mm512_add_pd(_mm512_mul_pd(px, z), p2);
	px = _mm512_mul_pd(px, x_clamped);

	qx = _mm512_add_pd(_mm512_mul_pd(q0, z), q1);
	qx = _mm512_add_pd(_mm512_mul_pd(qx, z), q2);
	qx = _mm512_add_pd(_mm512_mul_pd(qx, z), q3);

	y = _mm512_div_pd(px, _mm512_sub_pd(qx, px));
	y = _mm512_add_pd(one, _mm512_add_pd(y, y));
	y = _mm512_scalef_pd(y, n);

	y = _mm512_mask_mov_pd(y, underflow_mask, zero);
	y = _mm512_mask_mov_pd(y, overflow_mask, inf);
	y = _mm512_mask_mov_pd(y, nan_mask, x);
	return y;
}
#endif

#endif