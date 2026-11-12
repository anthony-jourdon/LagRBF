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
  file:    interpolation_avx512.c

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

#include <petsc.h>
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "cell_info.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx512.h"
#include <immintrin.h>

PetscLogEvent LAGRBF_PointwiseInterpolation_AVX512;
//extern inline __m512d KernelLinear_AVX512(__m512d r, __m512d epsilon);
//extern inline __m512d KernelWendlandC2_AVX512(__m512d r, __m512d epsilon);

static inline __m512d ComputeDistance_AVX512_2D(
  PetscReal *coords_i, 
  PetscReal *coords_d, 
  PetscInt np, 
  PetscInt *ci_data_idx, 
  PetscInt nd
)
{
  __m512d dx,dy,dist2,dist;
  /* Compute distances for this chunk of 8 points */
  dx = _mm512_set_pd(
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+7] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+6] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+5] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+4] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+3] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+2] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+1] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+0] ]
  );
  dy = _mm512_set_pd(
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+7] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+6] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+5] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+4] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+3] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+2] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+1] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+0] + 1 ]
  );
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_mul_pd(dy, dy)); // dx*dx + dy*dy
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

static inline __m512d ComputeDistance_AVX512_3D(
  PetscReal *coords_i, 
  PetscReal *coords_d, 
  PetscInt np, 
  PetscInt *ci_data_idx, 
  PetscInt nd
)
{
  __m512d dx,dy,dz,dist2,dist;
  /* Compute distances for this chunk of 8 points */
  dx = _mm512_set_pd(
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+7] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+6] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+5] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+4] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+3] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+2] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+1] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+0] ]
  );
  dy = _mm512_set_pd(
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+7] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+6] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+5] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+4] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+3] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+2] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+1] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+0] + 1 ]
  );
  dz = _mm512_set_pd(
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+7] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+6] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+5] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+4] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+3] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+2] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+1] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+0] + 2 ]
  );
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz))); // dx*dx + dy*dy + dz*dz
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode PointwiseInterpolation_AVX512(
  PartitionOfUnity pu,
  Partition        *partitions,
  CellInfo         *cellinfo,
  KernelType       ktype,
  PetscInt         nfields,
  PetscInt         npoints_i,
  PetscInt         *cellid_i,
  PetscReal        *coords_i,
  PetscReal        *coords_d,
  PetscReal        *rbf_d,
  PetscReal        *weights,
  PetscReal        *values_i
)
{
  DM             da;
  PetscInt       dim,k,np,nel,nen;
  PetscInt       *gtol_elidx;
  const PetscInt *elidx;
  PetscReal      *interpolant;
  __m512d        epsilon;
  __m512d        *interpolant_vec;
  __m512d        (*compute_distance)(PetscReal*,PetscReal*,PetscInt,PetscInt*,PetscInt) = NULL;
  __m512d        (*kernel_avx)(__m512d,__m512d) = NULL;
  PetscReal      (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;
  /* Logging */
  PetscCall(PetscLogEventBegin(LAGRBF_PointwiseInterpolation_AVX512,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Select the appropriate kernel */
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  PetscCall(KernelGetFunction_AVX512(ktype,&kernel_avx));
  /* Get parallel-sequential elements mapping */
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  interpolant_vec = (__m512d *)_mm_malloc(nfields * sizeof(__m512d), 64);
  /* Epsilon for the RBF kernel */
  epsilon = _mm512_set1_pd(pu->epsilon_r);
  /* Select the appropriate distance computation function */
  switch (dim) {
    case 2:
      compute_distance = ComputeDistance_AVX512_2D;
      break;
    case 3:
      compute_distance = ComputeDistance_AVX512_3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT " for AVX512 interpolation",dim);
  }
  for (np=0; np<npoints_i; np++) {
    CellInfo ci;
    PetscInt i,f,nd,cidx,local_cidx,ci_ndata,nd_aligned;
    PetscInt *ci_data_idx;

    cidx       = cellid_i[np];     // parallel DMDA cell index for the particle p
    local_cidx = gtol_elidx[cidx]; // local sequential DMDA cell index for the particle p
    /* Cell info of this cell */
    ci = cellinfo[local_cidx];
    PetscCall(CellInfoGetDataPointsIdx(ci,&ci_ndata,&ci_data_idx));
    if (ci_ndata == 0) continue; // no data points in this cell neighbourhood, skip
    /* Precompute the RBF values for all data points connected to that cell */
    nd_aligned = (ci_ndata / 8) * 8;
    for (nd=0; nd<nd_aligned; nd+=8) {
      double  rbf_arr[8] __attribute__((aligned(64)));
      __m512d dist,rbf;
      dist = compute_distance(coords_i, coords_d, np, ci_data_idx, nd);
      rbf  = kernel_avx(dist, epsilon);
      /* Store rbf values */
      _mm512_store_pd(rbf_arr, rbf);
      for (i=0; i<8; i++) {
        rbf_d[ ci_data_idx[nd + i] ] = (PetscReal)rbf_arr[i];
      }
    }
    /* remaining data points that are not a multiple of 8 */
    for (nd=nd_aligned; nd<ci_ndata; nd++) {
      PetscInt  d,dpidx;
      PetscReal dist;
      dpidx = ci_data_idx[nd];
      /* Compute the RBF value for the interpolation point */
      dist = 0.0;
      for (d=0; d<dim; d++) {
        PetscReal dx = coords_i[dim*np + d] - coords_d[dim*dpidx + d];
        dist += dx * dx;
      }
      dist = PetscSqrtReal(dist);
      rbf_d[dpidx] = kernel_fn(dist, pu->epsilon_r);
    }
    /* Initialize the interpolation value for the point p */
    PetscCall(PetscArrayzero(&values_i[nfields*np],nfields));
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  puidx,ndata;
      PetscInt  *data_idx;
      PetscReal  weight;
      PetscReal *coeffs;

      /* Get the partition index */
      puidx = elidx[nen*local_cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      /* Initialize the interpolant */
      for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm512_setzero_pd(); }
      /* Get the number of data points by chunk of 8 */
      nd_aligned = (ndata / 8) * 8;
      for (nd=0; nd<nd_aligned; nd+=8) {
        /* Compute the dot product R^T * X for each field */
        __m512d rbf;
        rbf = _mm512_set_pd(
          rbf_d[ data_idx[nd+7] ],
          rbf_d[ data_idx[nd+6] ],
          rbf_d[ data_idx[nd+5] ],
          rbf_d[ data_idx[nd+4] ],
          rbf_d[ data_idx[nd+3] ],
          rbf_d[ data_idx[nd+2] ],
          rbf_d[ data_idx[nd+1] ],
          rbf_d[ data_idx[nd+0] ]
        );
        for (f=0; f<nfields; f++) {
          __m512d coeff_v;
          // Need to check how to get coefficients aligned
          coeff_v = _mm512_loadu_pd(&coeffs[f * ndata + nd]);
          //coeff_v = _mm512_load_pd(&coeffs[f * ndata + nd]);
          interpolant_vec[f] = _mm512_fmadd_pd(rbf, coeff_v, interpolant_vec[f]);
        }
      }
      /* Extract the chunk of 8 - horizontal sum of 8 doubles */
      for (f=0; f<nfields; f++) {
        __m256d hi256,lo256,sum256;
        __m128d hi128,lo128,sum128,final;
        hi256  = _mm512_extractf64x4_pd(interpolant_vec[f], 1);
        lo256  = _mm512_castpd512_pd256(interpolant_vec[f]);
        sum256 = _mm256_add_pd(lo256, hi256);
        /* Now sum the 4 elements of the 256-bit result */
        hi128  = _mm256_extractf128_pd(sum256, 1);
        lo128  = _mm256_castpd256_pd128(sum256);
        sum128 = _mm_add_pd(lo128, hi128);
        final  = _mm_hadd_pd(sum128, sum128);
        interpolant[f] = _mm_cvtsd_f64(final);
      }
      /* remaining data points that are not a multiple of 4 */
      for (nd=nd_aligned; nd<ndata; nd++) {
        PetscReal rbf = rbf_d[ data_idx[nd] ];
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      /* Compute the global interpolation */
      weight = weights[nen*np + k];
      for (f=0; f<nfields; f++) {
        values_i[nfields*np + f] += interpolant[f] * weight;
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant));
  _mm_free(interpolant_vec);
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscCall(PetscLogEventEnd(LAGRBF_PointwiseInterpolation_AVX512,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}