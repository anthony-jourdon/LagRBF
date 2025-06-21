#include <petsc.h>

#include "partition_of_unity.h"
#include "partition.h"
#include "kernels_rbf.h"
#include "partition_solve.h"

PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix;
PetscLogEvent LAGRBF_PartitionFactorRBFMatrix;
PetscLogEvent LAGRBF_PartitionMatSetRHS;
PetscLogEvent LAGRBF_PartitionSolveLocalCoefficients;
PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;

static PetscErrorCode PartitionCreateRBFMatrix(Partition p)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  PetscCheck(!p->A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix already created.");
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&p->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionAssembleRBFMatrix(
  Partition p, 
  PetscInt dim, 
  PetscReal coords[], 
  PetscReal (*kernel_fn)(PetscReal,PetscReal)
)
{
  Mat       A;
  PetscInt  i,j,d,npoints;
  PetscInt  *pidx;
  PetscReal epsilon;
  PetscReal *rbf;
  PetscFunctionBeginUser;

  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix,0,0,0,0));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(PartitionGetEpsilon(p,&epsilon));
  PetscCall(MatDenseGetArray(A,&rbf));
  /* Fill the matrix */
  for (i=0; i<npoints; i++) {
    for (j=i; j<npoints; j++) {
      PetscReal dist = 0.0;
      /* Compute the distance between points */
      for (d=0; d<dim; d++) {
        dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
      }
      dist = PetscSqrtReal(dist);
      /* Fill the upper triangular part with the RBF kernel */
      rbf[npoints*i + j] = kernel_fn(dist,epsilon);
      /* Copy the value to the lower triangular part (diagonal already filled) */
      if (i != j) {
        rbf[npoints*j + i] = rbf[npoints*i + j];
      }
    }
  }
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionFactorRBFMatrix(Partition p)
{
  MatFactorInfo factinfo;
  IS            perm,cperm;
  Mat           A;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionFactorRBFMatrix,0,0,0,0));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &perm, &cperm));
  PetscCall(ISDestroy(&cperm));
  /* initialize factinfo */
  PetscCall(MatFactorInfoInitialize(&factinfo));
  /* factor */
  PetscCall(MatCholeskyFactor(A,perm,&factinfo));
  PetscCall(ISDestroy(&perm));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionFactorRBFMatrix,0,0,0,0));
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
  PetscCall(PartitionCreateRBFMatrix(p));
  PetscCall(PartitionAssembleRBFMatrix(p,dim,coords,kernel_fn));
  PetscCall(PartitionFactorRBFMatrix(p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B)
{
  Mat       RHS;
  PetscInt  f,np,npoints;
  PetscInt  *idx;
  PetscReal *LA_rhs;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionMatSetRHS,0,0,0,0));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&idx));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,nfields,NULL,&RHS));
  PetscCall(MatDenseGetArray(RHS,&LA_rhs));
  /* 
    Fill the matrix 
    WARNING: The array accessed by MatDenseGetArray() is stored column by column (not the usual way)

      -----> fields 
    | x00 x01 ... x0f |     
    | x10 x11 ... x1f |    [npoints of field 0][npoints of field 1] ... [npoints of field f] 
    | x20 x21 ... x2f | ==> x00 x10 x20 ... x0n x01 x11 x21 ... xn1 ... x0f x1f x2f ... xnf
    | ... ... ... ... |
    | xn0 xn1 ... xnf |

  */
  for (np=0; np<npoints; np++) {
    for (f=0; f<nfields; f++) {
      LA_rhs[f*npoints + np] = fval[ idx[np]*nfields + f ]; // set the RHS values
    }
  }
  PetscCall(MatDenseRestoreArray(RHS,&LA_rhs));
  *B = RHS;
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionMatSetRHS,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionMatSetRHS_Columnwise(Partition p, PetscInt nfields, PetscReal fval[], Mat *B)
{
  Mat       RHS;
  PetscInt  f,np,npoints;
  PetscInt  *idx;
  PetscReal *LA_rhs;
  PetscFunctionBeginUser;
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&idx));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,nfields,NULL,&RHS));
  for (f=0; f<nfields; f++) {
    PetscCall(MatDenseGetColumn(RHS,f,&LA_rhs));
    for (np=0; np<npoints; np++) {
      LA_rhs[np] = fval[ idx[np]*nfields + f ]; // set the RHS values
    }
    PetscCall(MatDenseRestoreColumn(RHS,&LA_rhs));
  }
  *B = RHS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionSolveLocalCoefficients(Partition p, PetscInt nfields, PetscReal fval[])
{
  Mat      A,B,X;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionSolveLocalCoefficients,0,0,0,0));
  PetscCall(PartitionGetCoefficientsMat(p,&X));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionMatSetRHS(p,nfields,fval,&B));
  /* solve the linear system A*X = B */
  PetscCall(MatMatSolve(A,B,X));
  /* Clean-up */
  PetscCall(MatDestroy(&B));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionSolveLocalCoefficients,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSolveInterpolationCoefficients(PartitionOfUnity pu, Partition *partitions, DM swarm)
{
  PetscInt  n,npart,nfields;
  PetscReal *values;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionSolveInterpolationCoefficients,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(DMSwarmGetField(swarm,"fields",&nfields,NULL,(void **)&values));
  for (n=0; n<npart; n++) {
    Partition p = partitions[n];
    PetscInt  npoints;
    PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
    if (npoints == 0) continue; // no data points in this partition, skip
    PetscCall(PartitionSolveLocalCoefficients(p,nfields,values));
  }
  /* Restore the field */
  PetscCall(DMSwarmRestoreField(swarm,"fields",&nfields,NULL,(void **)&values));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionSolveInterpolationCoefficients,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}