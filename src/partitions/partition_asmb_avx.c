#include <immintrin.h>
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_AVX256;

static inline __m256d ComputeDistance_AVX256_2D(PetscReal *coords, PetscInt *idx, __m256d *xi)
{
  __m256d xj,yj,dx,dy,dist2,dist;
  /* Load 4 points */
  xj = _mm256_set_pd(
    coords[2*idx[3]],
    coords[2*idx[2]],
    coords[2*idx[1]],
    coords[2*idx[0]]
  );
  yj = _mm256_set_pd(
    coords[2*idx[3] + 1],
    coords[2*idx[2] + 1],
    coords[2*idx[1] + 1],
    coords[2*idx[0] + 1]
  );
  
  /* Compute distances */
  dx = _mm256_sub_pd(xi[0], xj);
  dy = _mm256_sub_pd(xi[1], yj);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
  dist = _mm256_sqrt_pd(dist2);
  return dist;
}

static inline __m256d ComputeDistance_AVX256_3D(PetscReal *coords, PetscInt *idx, __m256d *xi)
{
  __m256d xj,yj,zj,dx,dy,dz,dist2,dist;
  /* Load 4 points */
  xj = _mm256_set_pd(
    coords[3*idx[3]],
    coords[3*idx[2]],
    coords[3*idx[1]],
    coords[3*idx[0]]
  );
  yj = _mm256_set_pd(
    coords[3*idx[3] + 1],
    coords[3*idx[2] + 1],
    coords[3*idx[1] + 1],
    coords[3*idx[0] + 1]
  );
  zj = _mm256_set_pd(
    coords[3*idx[3] + 2],
    coords[3*idx[2] + 2],
    coords[3*idx[1] + 2],
    coords[3*idx[0] + 2]
  );
  
  /* Compute distances */
  dx = _mm256_sub_pd(xi[0], xj);
  dy = _mm256_sub_pd(xi[1], yj);
  dz = _mm256_sub_pd(xi[2], zj);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
  dist = _mm256_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode PartitionAssembleRBFMatrix_AVX256(
  Partition  p,
  PetscInt   dim,
  PetscReal  epsilon,
  PetscReal *coords,
  KernelType ktype
)
{
  Mat       A;
  PetscInt  i,j,k,d,npoints;
  PetscInt  *pidx;
  PetscReal *rbf;
  __m256d   eps_vec;
  __m256d   (*compute_distance)(PetscReal *, PetscInt *, __m256d *) = NULL;
  __m256d   (*kernel_avx)(__m256d,__m256d) = NULL;
  PetscReal (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix_AVX256,0,0,0,0));

  /* Get the AVX kernel function */
  PetscCall(KernelGetFunction_AVX256(ktype, &kernel_avx));
  PetscCall(KernelGetFunction(ktype,&kernel_fn));

  switch(dim) {
    case 2:
      compute_distance = ComputeDistance_AVX256_2D;
      break;
    case 3:
      compute_distance = ComputeDistance_AVX256_3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "AVX optimization only supports 2D and 3D, got dim = %" PetscInt_FMT, dim);
  }
  
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(MatDenseGetArray(A,&rbf));
  
  eps_vec = _mm256_set1_pd(epsilon);
  for (i=0; i<npoints; i++) {
    __m256d xi[3]; // Max 3 dimensions
    for (d=0; d<dim; d++) {
      /* Broadcast the i-th point's coordinate in dimension d */
      xi[d] = _mm256_set1_pd(coords[dim*pidx[i] + d]);
    }

    /* Process 4 columns at a time */
    for (j=i; j<(npoints-3); j+=4) {
      double kernel_arr[4] __attribute__((aligned(32)));
      __m256d dist, kernel_val;
      
      /* Compute distance */
      dist       = compute_distance(coords, &pidx[j], xi);
      /* Apply kernel */
      kernel_val = kernel_avx(dist, eps_vec);
      _mm256_store_pd(kernel_arr, kernel_val);
      
      /* Store in matrix (upper triangle) */
      for (k=0; k<4; k++) {
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
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/* AVX-optimized version of PartitionAssembleRBFMatrix for 2D */
static PetscErrorCode PartitionAssembleRBFMatrix_AVX_2D(
  Partition p,
  PetscReal epsilon,
  PetscReal coords[],
  PetscReal (*kernel_fn)(PetscReal,PetscReal),
  __m256d (*kernel_avx)(__m256d,__m256d)
)
{
  Mat       A;
  PetscInt  i,j,npoints;
  PetscInt  *pidx;
  PetscReal *rbf;
  __m256d   eps_vec;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(MatDenseGetArray(A,&rbf));
  
  eps_vec = _mm256_set1_pd(epsilon);
  
  /* Fill the matrix using AVX */
  for (i=0; i<npoints; i++) {
    PetscReal xi = coords[2*pidx[i]];
    PetscReal yi = coords[2*pidx[i] + 1];
    __m256d xi_vec = _mm256_set1_pd(xi);
    __m256d yi_vec = _mm256_set1_pd(yi);
    
    /* Process 4 columns at a time */
    for (j=i; j<(npoints-3); j+=4) {
      double dist_arr[4] __attribute__((aligned(32)));
      double kernel_arr[4] __attribute__((aligned(32)));
      __m256d xj,yj,dx,dy,dist2,dist,kernel_val;
      
      /* Load 4 points */
      xj = _mm256_set_pd(
        coords[2*pidx[j+3]],
        coords[2*pidx[j+2]],
        coords[2*pidx[j+1]],
        coords[2*pidx[j]]
      );
      yj = _mm256_set_pd(
        coords[2*pidx[j+3] + 1],
        coords[2*pidx[j+2] + 1],
        coords[2*pidx[j+1] + 1],
        coords[2*pidx[j] + 1]
      );
      
      /* Compute distances */
      dx = _mm256_sub_pd(xi_vec, xj);
      dy = _mm256_sub_pd(yi_vec, yj);
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
      dist = _mm256_sqrt_pd(dist2);
      
      /* Apply kernel */
      kernel_val = kernel_avx(dist, eps_vec);
      _mm256_store_pd(kernel_arr, kernel_val);
      
      /* Store in matrix (upper triangle) */
      for (PetscInt k=0; k<4; k++) {
        rbf[npoints*i + j + k] = (PetscReal)kernel_arr[k];
        /* Mirror to lower triangle */
        if (i != j+k) {
          rbf[npoints*(j+k) + i] = (PetscReal)kernel_arr[k];
        }
      }
    }
    
    /* Handle remaining columns */
    for (; j<npoints; j++) {
      PetscReal dx = xi - coords[2*pidx[j]];
      PetscReal dy = yi - coords[2*pidx[j] + 1];
      PetscReal dist = PetscSqrtReal(dx*dx + dy*dy);
      PetscReal kernel_val = kernel_fn(dist, epsilon);
      rbf[npoints*i + j] = kernel_val;
      if (i != j) {
        rbf[npoints*j + i] = kernel_val;
      }
    }
  }
  
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AVX-optimized version of PartitionAssembleRBFMatrix for 3D */
static PetscErrorCode PartitionAssembleRBFMatrix_AVX_3D(
  Partition p,
  PetscReal epsilon,
  PetscReal coords[],
  PetscReal (*kernel_fn)(PetscReal,PetscReal),
  __m256d (*kernel_avx)(__m256d,__m256d)
)
{
  Mat       A;
  PetscInt  i,j,npoints;
  PetscInt  *pidx;
  PetscReal *rbf;
  __m256d   eps_vec;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(MatDenseGetArray(A,&rbf));
  
  eps_vec = _mm256_set1_pd(epsilon);
  
  /* Fill the matrix using AVX */
  for (i=0; i<npoints; i++) {
    PetscReal xi = coords[3*pidx[i]];
    PetscReal yi = coords[3*pidx[i] + 1];
    PetscReal zi = coords[3*pidx[i] + 2];
    __m256d xi_vec = _mm256_set1_pd(xi);
    __m256d yi_vec = _mm256_set1_pd(yi);
    __m256d zi_vec = _mm256_set1_pd(zi);
    
    /* Process 4 columns at a time */
    for (j=i; j<(npoints-3); j+=4) {
      double kernel_arr[4] __attribute__((aligned(32)));
      __m256d xj,yj,zj,dx,dy,dz,dist2,dist,kernel_val;
      
      /* Load 4 points */
      xj = _mm256_set_pd(
        coords[3*pidx[j+3]],
        coords[3*pidx[j+2]],
        coords[3*pidx[j+1]],
        coords[3*pidx[j]]
      );
      yj = _mm256_set_pd(
        coords[3*pidx[j+3] + 1],
        coords[3*pidx[j+2] + 1],
        coords[3*pidx[j+1] + 1],
        coords[3*pidx[j] + 1]
      );
      zj = _mm256_set_pd(
        coords[3*pidx[j+3] + 2],
        coords[3*pidx[j+2] + 2],
        coords[3*pidx[j+1] + 2],
        coords[3*pidx[j] + 2]
      );
      
      /* Compute distances */
      dx = _mm256_sub_pd(xi_vec, xj);
      dy = _mm256_sub_pd(yi_vec, yj);
      dz = _mm256_sub_pd(zi_vec, zj);
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
      dist = _mm256_sqrt_pd(dist2);
      
      /* Apply kernel */
      kernel_val = kernel_avx(dist, eps_vec);
      _mm256_store_pd(kernel_arr, kernel_val);
      
      /* Store in matrix (upper triangle) */
      for (PetscInt k=0; k<4; k++) {
        rbf[npoints*i + j + k] = (PetscReal)kernel_arr[k];
        /* Mirror to lower triangle */
        if (i != j+k) {
          rbf[npoints*(j+k) + i] = (PetscReal)kernel_arr[k];
        }
      }
    }
    
    /* Handle remaining columns */
    for (; j<npoints; j++) {
      PetscReal dx = xi - coords[3*pidx[j]];
      PetscReal dy = yi - coords[3*pidx[j] + 1];
      PetscReal dz = zi - coords[3*pidx[j] + 2];
      PetscReal dist = PetscSqrtReal(dx*dx + dy*dy + dz*dz);
      PetscReal kernel_val = kernel_fn(dist, epsilon);
      rbf[npoints*i + j] = kernel_val;
      if (i != j) {
        rbf[npoints*j + i] = kernel_val;
      }
    }
  }
  
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Wrapper function that selects AVX or scalar version */
PetscErrorCode PartitionAssembleRBFMatrix_Optimized(
  Partition p,
  PetscInt dim,
  PetscReal epsilon,
  PetscReal coords[],
  KernelType ktype
)
{
  __m256d (*kernel_avx)(__m256d,__m256d) = NULL;
  PetscReal (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;

  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix_AVX256,0,0,0,0));
  
  /* Get the AVX kernel function */
  PetscCall(KernelGetFunction_AVX256(ktype, &kernel_avx));
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  
  /* Call dimension-specific AVX version */
  if (dim == 2) {
    PetscCall(PartitionAssembleRBFMatrix_AVX_2D(p, epsilon, coords, kernel_fn, kernel_avx));
  } else if (dim == 3) {
    PetscCall(PartitionAssembleRBFMatrix_AVX_3D(p, epsilon, coords, kernel_fn, kernel_avx));
  } else {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "AVX optimization only supports 2D and 3D, got dim=%d", dim);
  }
  
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif