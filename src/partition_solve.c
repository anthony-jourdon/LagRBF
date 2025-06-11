#include <petsc.h>

#include "partition_of_unity.h"
#include "partition.h"
#include "kernels_rbf.h"
#include "partition_solve.h"

PetscErrorCode PartitionAssembleRBFMatrix(
  Partition p, 
  PetscInt dim, 
  PetscReal coords[], 
  PetscReal (*kernel_fn)(PetscReal,PetscReal), 
  Mat *A
)
{
  Mat            RBF;
  PetscInt       i,j,d,npoints;
  PetscInt       *pidx,*mat_idx;
  PetscReal      epsilon;
  PetscReal      *rbf;
  PetscFunctionBeginUser;

  /* Get the data points belonging to this partition */
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PartitionGetEpsilon(p,&epsilon);
  /* Create the rbf matrix to solve for local interpolation coefficients */
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,npoints,NULL,&RBF));
  PetscCall(MatSetOptionsPrefix(RBF,"rbf_"));
  PetscCall(MatSetFromOptions(RBF));
  // MatDenseGetArray(Mat A, PetscScalar *array[]) -> USE THIS DIRECTLY TO GET THE ARRAY
  // Fill lower (or upper) triangular part of the matrix and copy to the upper (or lower) triangular part from the other part
  /* allocate matrix array and matrix indices */
  PetscMalloc2(npoints*npoints,&rbf,npoints,&mat_idx);
  /* compute the rbf matrix */
  for (i=0; i<npoints; i++){
    mat_idx[i] = i;
    for (j=0; j<npoints; j++) {
      PetscReal dist = 0.0;
      for (d=0; d<dim; d++) {
        dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
      }
      dist = PetscSqrtReal(dist);
      rbf[npoints*i + j] = kernel_fn(dist,epsilon);
    }
  }
  /* Set the values in the matrix */
  PetscCall(MatSetValues(RBF,npoints,mat_idx,npoints,mat_idx,rbf,ADD_VALUES));
  PetscCall(MatAssemblyBegin(RBF,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RBF,MAT_FINAL_ASSEMBLY));
  /* Set the output matrix */
  *A = RBF;
  /* Clean-up */
  PetscCall(PetscFree(rbf));
  PetscCall(PetscFree(mat_idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  We want to interpolate different fields, therefore it is required to 
  solve the system Ax = b as much as there are fields to interpolate.
  However, the matrix A is the same for all fields, so we can factorize it once
  and reuse the factorization for each field. 
*/
PetscErrorCode PartitionFactorRBFMatrix(
  Partition p, 
  PetscInt dim, 
  PetscReal coords[],
  PetscReal (*kernel_fn)(PetscReal,PetscReal),
  Mat *A
)
{
  MatFactorInfo factinfo;
  Mat           RBF;
  IS            perm,cperm;
  
  PetscFunctionBeginUser;
  /* Fill the RBF matrix */
  PetscCall(PartitionAssembleRBFMatrix(p,dim,coords,kernel_fn,&RBF));
  PetscCall(MatGetOrdering(RBF, MATORDERINGNATURAL, &perm, &cperm));
  PetscCall(ISDestroy(&cperm));
  /* initialize factinfo */
  PetscCall(MatFactorInfoInitialize(&factinfo));
  /* factor */
  PetscCall(MatCholeskyFactor(RBF,perm,&factinfo));
  PetscCall(ISDestroy(&perm));
  *A = RBF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetUpRBFMatrix(
  Partition p, 
  PetscInt dim, 
  PetscReal coords[],
  PetscReal (*kernel_fn)(PetscReal,PetscReal)
)
{
  PetscFunctionBeginUser;
  PartitionFactorRBFMatrix(p,dim,coords,kernel_fn,&p->A);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetVecRHS(Partition p, PetscReal fvals[], Vec *B)
{
  Vec       RHS;
  PetscInt  i,npoints;
  PetscInt  *idx;
  PetscReal *rhs;
  PetscFunctionBeginUser;
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&idx));
  PetscCall(VecCreate(PETSC_COMM_SELF,&RHS));
  PetscCall(VecSetSizes(RHS,PETSC_DECIDE,npoints));
  PetscCall(VecSetFromOptions(RHS));
  PetscCall(VecGetArray(RHS,&rhs));
  /* set the right-hand side with the values of the function at the data points */
  for (i=0; i<npoints; i++){ 
    rhs[i] = fvals[ idx[i] ]; 
  }
  PetscCall(VecRestoreArray(RHS,&rhs));
  *B = RHS;
  PetscFunctionReturn(PETSC_SUCCESS);
}