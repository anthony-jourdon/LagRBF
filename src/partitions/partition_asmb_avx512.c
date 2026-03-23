#include <immintrin.h>
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx512.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_AVX512;
extern inline __m512d KernelLinear_AVX512(__m512d r, __m512d epsilon);
extern inline __m512d KernelWendlandC0_AVX512(__m512d r, __m512d epsilon);
extern inline __m512d KernelWendlandC2_AVX512(__m512d r, __m512d epsilon);
extern inline __m512d KernelBump_AVX512(__m512d r, __m512d epsilon);

static inline __m512d ComputeDistance_AVX512_2D(PetscReal *coords, PetscInt *idx, __m512d *xi)
{
  __m512d xj,yj,dx,dy,dist2,dist;
  /* Load 8 points */
  xj = _mm512_set_pd(
    coords[2*idx[7]],
    coords[2*idx[6]],
    coords[2*idx[5]],
    coords[2*idx[4]],
    coords[2*idx[3]],
    coords[2*idx[2]],
    coords[2*idx[1]],
    coords[2*idx[0]]
  );
  yj = _mm512_set_pd(
    coords[2*idx[7] + 1],
    coords[2*idx[6] + 1],
    coords[2*idx[5] + 1],
    coords[2*idx[4] + 1],
    coords[2*idx[3] + 1],
    coords[2*idx[2] + 1],
    coords[2*idx[1] + 1],
    coords[2*idx[0] + 1]
  );
  
  /* Compute distances */
  dx = _mm512_sub_pd(xi[0], xj);
  dy = _mm512_sub_pd(xi[1], yj);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_mul_pd(dy, dy));
  dist = _mm512_sqrt_pd(dist2);
  return dist;
}

static inline __m512d ComputeDistance_AVX512_3D(PetscReal *coords, PetscInt *idx, __m512d *xi)
{
  __m512d xj,yj,zj,dx,dy,dz,dist2,dist;
  /* Load 8 points */
  xj = _mm512_set_pd(
    coords[3*idx[7]],
    coords[3*idx[6]],
    coords[3*idx[5]],
    coords[3*idx[4]],
    coords[3*idx[3]],
    coords[3*idx[2]],
    coords[3*idx[1]],
    coords[3*idx[0]]
  );
  yj = _mm512_set_pd(
    coords[3*idx[7] + 1],
    coords[3*idx[6] + 1],
    coords[3*idx[5] + 1],
    coords[3*idx[4] + 1],
    coords[3*idx[3] + 1],
    coords[3*idx[2] + 1],
    coords[3*idx[1] + 1],
    coords[3*idx[0] + 1]
  );
  zj = _mm512_set_pd(
    coords[3*idx[7] + 2],
    coords[3*idx[6] + 2],
    coords[3*idx[5] + 2],
    coords[3*idx[4] + 2],
    coords[3*idx[3] + 2],
    coords[3*idx[2] + 2],
    coords[3*idx[1] + 2],
    coords[3*idx[0] + 2]
  );
  
  /* Compute distances */
  dx = _mm512_sub_pd(xi[0], xj);
  dy = _mm512_sub_pd(xi[1], yj);
  dz = _mm512_sub_pd(xi[2], zj);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
  dist = _mm512_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode PartitionAssembleRBFMatrix_AVX512(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype)
{
  Mat       A;
  PetscInt  i,j,k,d,npoints;
  PetscInt  *pidx;
  PetscReal *rbf;
  __m512d   eps_vec;
  __m512d   (*compute_distance)(PetscReal *, PetscInt *, __m512d *) = NULL;
  __m512d   (*kernel_avx)(__m512d,__m512d) = NULL;
  PetscReal (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix_AVX512,0,0,0,0));

  /* Get the AVX512 kernel function */
  PetscCall(KernelGetFunction_AVX512(ktype, &kernel_avx));
  PetscCall(KernelGetFunction(ktype,&kernel_fn));

  switch(dim) {
    case 2:
      compute_distance = ComputeDistance_AVX512_2D;
      break;
    case 3:
      compute_distance = ComputeDistance_AVX512_3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "AVX512 optimization only supports 2D and 3D, got dim = %" PetscInt_FMT, dim);
  }
  
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(MatDenseGetArray(A,&rbf));
  
  eps_vec = _mm512_set1_pd(epsilon);
  for (i=0; i<npoints; i++) {
    __m512d xi[3]; // Max 3 dimensions
    for (d=0; d<dim; d++) {
      /* Broadcast the i-th point's coordinate in dimension d */
      xi[d] = _mm512_set1_pd(coords[dim*pidx[i] + d]);
    }

    /* Process 8 columns at a time */
    for (j=i; j<(npoints-7); j+=8) {
      double kernel_arr[8] __attribute__((aligned(64)));
      __m512d dist, kernel_val;
      
      /* Compute distance */
      dist       = compute_distance(coords, &pidx[j], xi);
      /* Apply kernel */
      kernel_val = kernel_avx(dist, eps_vec);
      _mm512_store_pd(kernel_arr, kernel_val);
      
      /* Store in matrix (upper triangle) */
      for (k=0; k<8; k++) {
        rbf[npoints*i + j + k] = (PetscReal)kernel_arr[k];
        /* Mirror to lower triangle */
        if (i != j+k) {
          rbf[npoints*(j+k) + i] = (PetscReal)kernel_arr[k];
        }
      }
    }
    /* Handle remaining columns */
    for (; j<npoints; j++) {
      PetscReal kernel_val;
      PetscReal dist = 0.0;
      /* Compute the distance between points */
      for (d=0; d<dim; d++) {
        dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
      }
      dist = PetscSqrtReal(dist);
      kernel_val = kernel_fn(dist, epsilon);
      rbf[npoints*i + j] = kernel_val;
      if (i != j) {
        rbf[npoints*j + i] = kernel_val;
      }
    }
  }
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix_AVX512,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}