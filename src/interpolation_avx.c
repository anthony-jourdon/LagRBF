#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"
#include "partition_solve.h"
#include "kernels_rbf.h"
#include <immintrin.h>

PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;

PetscErrorCode PointwiseInterpolation_AVX256_2D(
  PartitionOfUnity pu,
  Partition *partitions,
  PetscInt nfields,
  PetscInt npoints_i,
  PetscInt *cellid_i,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *weights,
  PetscReal *values_i
)
{
  DM             da;
  PetscInt       k,np,dim,npart,nel,nen;
  const PetscInt *elidx;
  PetscReal      *interpolant;
  
  __m256d        one,four,five;
  __m256d        *interpolant_vec;
  PetscFunctionBeginUser;
  
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  interpolant_vec = (__m256d *)_mm_malloc(nfields * sizeof(__m256d), 32);
  /* Set avx constants */
  one  = _mm256_set1_pd(1.0);
  four = _mm256_set1_pd(4.0);
  five = _mm256_set1_pd(5.0);
  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) { values_i[nfields*np + f] = 0.0; }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata,nd_aligned;
      PetscInt  *data_idx;
      PetscReal *coeffs;
      __m256d   epsilon;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      
      epsilon = _mm256_set1_pd(p->epsilon);
      /* Initialize the interpolant */
      for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm256_setzero_pd(); }
      /* Get the number of data points by chuck of 4 */
      nd_aligned = (ndata / 4) * 4;
      for (nd=0; nd<nd_aligned; nd+=4) {
        __m256d dx,dy,dist2,dist,u,mask,u2,u4,four_u,inner,rbf;
        // Step 1: Compute RBF values for this chunck of 4 points
        dx = _mm256_set_pd(
          coords_d[2*data_idx[nd+3] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+2] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+1] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+0] + 0] - coords_i[2*np + 0]
        );
        dy = _mm256_set_pd(
          coords_d[2*data_idx[nd+3] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+2] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+1] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+0] + 1] - coords_i[2*np + 1]
        );
        dist2  = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));    // dx*dx + dy*dy
        dist   = _mm256_sqrt_pd(dist2);                             // sqrt(dx*dx + dy*dy)
        u      = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, dist));  // u = 1 - epsilon * dist
        mask   = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
        u2     = _mm256_mul_pd(u, u);                               // u^2
        u4     = _mm256_mul_pd(u2, u2);                             // u^4
        four_u = _mm256_mul_pd(four, u);                            // 4 * u
        inner  = _mm256_sub_pd(five, four_u);                       // 5 - 4 * u
        rbf    = _mm256_mul_pd(u4, inner);                          // rbf = u^4 * (5 - 4 * u)
        rbf    = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask);  // apply mask to zero out values where u < 0
        
        /* Compute the dot product R^T * X for each field */
        for (f=0; f<nfields; f++) {
          __m256d coeff_v;
          coeff_v = _mm256_loadu_pd(&coeffs[f * ndata + nd]);
          interpolant_vec[f] = _mm256_fmadd_pd(rbf, coeff_v, interpolant_vec[f]);
        }
      }
      /* Extract the chuck of 4 */
      for (f=0; f<nfields; f++) {
        __m128d hi128,lo128,sum128,final;
        hi128  = _mm256_extractf128_pd(interpolant_vec[f], 1);
        lo128  = _mm256_castpd256_pd128(interpolant_vec[f]);
        sum128 = _mm_add_pd(lo128, hi128);
        final  = _mm_hadd_pd(sum128, sum128);
        interpolant[f] = _mm_cvtsd_f64(final);
      }
      /* remaining data points that are not a multiple of 4 */
      for (nd=nd_aligned; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          PetscReal dx = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += dx * dx;
        }
        dist = PetscSqrtReal(dist);
        rbf  = KernelWendlandC2(dist,p->epsilon);
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      /* Compute the global interpolation */
      for (f = 0; f < nfields; ++f) {
        values_i[nfields*np + f] += interpolant[f] * weights[np * npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant));
  _mm_free(interpolant_vec);
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PointwiseInterpolation_AVX256_3D(
  PartitionOfUnity pu,
  Partition *partitions,
  PetscInt nfields,
  PetscInt npoints_i,
  PetscInt *cellid_i,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *weights,
  PetscReal *values_i
)
{
  DM             da;
  PetscInt       k,np,dim,npart,nel,nen;
  const PetscInt *elidx;
  PetscReal      *interpolant;
  __m256d        one,four,five;
  __m256d        *interpolant_vec;
  PetscFunctionBeginUser;

  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  interpolant_vec = (__m256d *)_mm_malloc(nfields * sizeof(__m256d), 32);
  /* Set avx constants */
  one  = _mm256_set1_pd(1.0);
  four = _mm256_set1_pd(4.0);
  five = _mm256_set1_pd(5.0);
  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) { values_i[nfields*np + f] = 0.0; }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata,nd_aligned;
      PetscInt  *data_idx;
      PetscReal *coeffs;
      __m256d   epsilon;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      
      epsilon = _mm256_set1_pd(p->epsilon);
      /* Initialize the interpolant */
      for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm256_setzero_pd(); }
      /* Get the number of data points by chuck of 4 */
      nd_aligned = (ndata / 4) * 4;
      for (nd=0; nd<nd_aligned; nd+=4) {
        __m256d dx,dy,dz,dist2,dist,u,mask,u2,u4,four_u,inner,rbf;
        // Step 1: Compute RBF values for this chunck of 4 points
        dx = _mm256_set_pd(
          coords_d[3*data_idx[nd+3] + 0] - coords_i[3*np + 0],
          coords_d[3*data_idx[nd+2] + 0] - coords_i[3*np + 0],
          coords_d[3*data_idx[nd+1] + 0] - coords_i[3*np + 0],
          coords_d[3*data_idx[nd+0] + 0] - coords_i[3*np + 0]
        );
        dy = _mm256_set_pd(
          coords_d[3*data_idx[nd+3] + 1] - coords_i[3*np + 1],
          coords_d[3*data_idx[nd+2] + 1] - coords_i[3*np + 1],
          coords_d[3*data_idx[nd+1] + 1] - coords_i[3*np + 1],
          coords_d[3*data_idx[nd+0] + 1] - coords_i[3*np + 1]
        );
        dz = _mm256_set_pd(
          coords_d[3*data_idx[nd+3] + 2] - coords_i[3*np + 2],
          coords_d[3*data_idx[nd+2] + 2] - coords_i[3*np + 2],
          coords_d[3*data_idx[nd+1] + 2] - coords_i[3*np + 2],
          coords_d[3*data_idx[nd+0] + 2] - coords_i[3*np + 2]
        );
        // Compute the squared distance
        dist2  = _mm256_fmadd_pd(dx, dx, 
          _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));          // dx*dx + dy*dy + dz*dz
        dist   = _mm256_sqrt_pd(dist2);                             // sqrt(dx*dx + dy*dy + dz*dz)
        u      = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, dist));  // u = 1 - epsilon * dist
        mask   = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
        u2     = _mm256_mul_pd(u, u);                               // u^2
        u4     = _mm256_mul_pd(u2, u2);                             // u^4
        four_u = _mm256_mul_pd(four, u);                            // 4 * u
        inner  = _mm256_sub_pd(five, four_u);                       // 5 - 4 * u
        rbf    = _mm256_mul_pd(u4, inner);                          // rbf = u^4 * (5 - 4 * u)
        rbf    = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask);  // apply mask to zero out values where u < 0
        
        /* Compute the dot product R^T * X for each field */
        for (f=0; f<nfields; f++) {
          __m256d coeff_v;
          coeff_v = _mm256_loadu_pd(&coeffs[f * ndata + nd]);
          interpolant_vec[f] = _mm256_fmadd_pd(rbf, coeff_v, interpolant_vec[f]);
        }
      }
      /* Extract the chuck of 4 */
      for (f=0; f<nfields; f++) {
        __m128d hi128,lo128,sum128,final;
        hi128  = _mm256_extractf128_pd(interpolant_vec[f], 1);
        lo128  = _mm256_castpd256_pd128(interpolant_vec[f]);
        sum128 = _mm_add_pd(lo128, hi128);
        final  = _mm_hadd_pd(sum128, sum128);
        interpolant[f] = _mm_cvtsd_f64(final);
      }
      /* remaining data points that are not a multiple of 4 */
      for (nd=nd_aligned; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          PetscReal dx = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += dx * dx;
        }
        dist = PetscSqrtReal(dist);
        rbf  = KernelWendlandC2(dist,p->epsilon);
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      /* Compute the global interpolation */
      for (f = 0; f < nfields; ++f) {
        values_i[nfields*np + f] += interpolant[f] * weights[np * npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant));
  _mm_free(interpolant_vec);
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PointwiseInterpolation_AVX256(
  PartitionOfUnity pu,
  Partition *partitions,
  PetscInt nfields,
  PetscInt npoints_i,
  PetscInt *cellid_i,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *weights,
  PetscReal *values_i
)
{
  PetscInt       dim;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;

  /* Logging */
  PetscCall(PetscTime(&time[0]));
  PetscCall(PetscLogEventBegin(LAGRBF_PointwiseInterpolation_AVX256,0,0,0,0));
  
  PetscCall(PUGetDimension(pu,&dim));
  switch (dim) {
    case 2:
      PetscCall(PointwiseInterpolation_AVX256_2D(pu, partitions, nfields, npoints_i, cellid_i, coords_i, coords_d, weights, values_i));
      break;
    case 3:
      PetscCall(PointwiseInterpolation_AVX256_3D(pu, partitions, nfields, npoints_i, cellid_i, coords_i, coords_d, weights, values_i));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported dimension %d for AVX256 interpolation",dim);
  }
  
  PetscCall(PetscLogEventEnd(LAGRBF_PointwiseInterpolation_AVX256,0,0,0,0));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}