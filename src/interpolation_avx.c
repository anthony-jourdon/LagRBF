#include <petsc.h>
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "cell_info.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx.h"
#include <immintrin.h>

PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;
extern inline __m256d KernelLinear_AVX256(__m256d r, __m256d epsilon);
extern inline __m256d KernelWendlandC2_AVX256(__m256d r, __m256d epsilon);

static inline __m256d ComputeDistance_AVX256_2D(
  PetscReal *coords_i, 
  PetscReal *coords_d, 
  PetscInt np, 
  PetscInt *ci_data_idx, 
  PetscInt nd
)
{
  __m256d dx,dy,dist2,dist;
  /* Compute distances for this chunck of 4 points */
  dx = _mm256_set_pd(
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+3] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+2] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+1] ],
    coords_i[ 2*np ] - coords_d[ 2*ci_data_idx[nd+0] ]
  );
  dy = _mm256_set_pd(
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+3] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+2] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+1] + 1 ],
    coords_i[ 2*np + 1 ] - coords_d[ 2*ci_data_idx[nd+0] + 1 ]
  );
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy)); // dx*dx + dy*dy
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

static inline __m256d ComputeDistance_AVX256_3D(
  PetscReal *coords_i, 
  PetscReal *coords_d, 
  PetscInt np, 
  PetscInt *ci_data_idx, 
  PetscInt nd
)
{
  __m256d dx,dy,dz,dist2,dist;
  /* Compute distances for this chunck of 4 points */
  dx = _mm256_set_pd(
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+3] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+2] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+1] ],
    coords_i[ 3*np ] - coords_d[ 3*ci_data_idx[nd+0] ]
  );
  dy = _mm256_set_pd(
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+3] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+2] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+1] + 1 ],
    coords_i[ 3*np + 1 ] - coords_d[ 3*ci_data_idx[nd+0] + 1 ]
  );
  dz = _mm256_set_pd(
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+3] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+2] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+1] + 2 ],
    coords_i[ 3*np + 2 ] - coords_d[ 3*ci_data_idx[nd+0] + 2 ]
  );
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz))); // dx*dx + dy*dy + dz*dz
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode PointwiseInterpolation_AVX256(
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
  __m256d        epsilon;
  __m256d        *interpolant_vec;
  __m256d        (*compute_distance)(PetscReal*,PetscReal*,PetscInt,PetscInt*,PetscInt) = NULL;
  __m256d        (*kernel_avx)(__m256d,__m256d) = NULL;
  PetscReal      (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;
  /* Logging */
  PetscCall(PetscLogEventBegin(LAGRBF_PointwiseInterpolation_AVX256,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Select the appropriate kernel */
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  PetscCall(KernelGetFunction_AVX256(ktype,&kernel_avx));
  /* Get parallel-sequential elements mapping */
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  interpolant_vec = (__m256d *)_mm_malloc(nfields * sizeof(__m256d), 32);
  /* Epsilon for the RBF kernel */
  epsilon = _mm256_set1_pd(pu->epsilon);
  /* Select the appropriate distance computation function */
  switch (dim) {
    case 2:
      compute_distance = ComputeDistance_AVX256_2D;
      break;
    case 3:
      compute_distance = ComputeDistance_AVX256_3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT " for AVX256 interpolation",dim);
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
    nd_aligned = (ci_ndata / 4) * 4;
    for (nd=0; nd<nd_aligned; nd+=4) {
      double  rbf_arr[4] __attribute__((aligned(32)));
      __m256d dist,rbf;
      dist = compute_distance(coords_i, coords_d, np, ci_data_idx, nd);
      rbf  = kernel_avx(dist, epsilon);
      /* Store rbf values */
      _mm256_store_pd(rbf_arr, rbf);
      for (i=0; i<4; i++) {
        rbf_d[ ci_data_idx[nd + i] ] = (PetscReal)rbf_arr[i];
      }
    }
    /* remaining data points that are not a multiple of 4 */
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
      rbf_d[dpidx] = kernel_fn(dist, pu->epsilon);
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
      //for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm256_setzero_pd(); }
      memset(interpolant_vec, 0, nfields * sizeof(__m256d));
      /* Get the number of data points by chuck of 4 */
      nd_aligned = (ndata / 4) * 4;
      for (nd=0; nd<nd_aligned; nd+=4) {
        /* Compute the dot product R^T * X for each field */
        __m256d rbf;
        rbf = _mm256_set_pd(
          rbf_d[ data_idx[nd+3] ],
          rbf_d[ data_idx[nd+2] ],
          rbf_d[ data_idx[nd+1] ],
          rbf_d[ data_idx[nd+0] ]
        );
        for (f=0; f<nfields; f++) {
          __m256d coeff_v;
          // Need to check how to get coefficients aligned
          coeff_v = _mm256_loadu_pd(&coeffs[f * ndata + nd]);
          //coeff_v = _mm256_load_pd(&coeffs[f * ndata + nd]);
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
  PetscCall(PetscLogEventEnd(LAGRBF_PointwiseInterpolation_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}