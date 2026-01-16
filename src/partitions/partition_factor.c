#include <petsc.h>
#include "partitions/partition.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_FactorRBFMatrix;
PetscLogEvent LAGRBF_MatSetRHS_Fac;
PetscLogEvent LAGRBF_SolveLocalCoefficients_Fac;

static PetscErrorCode PartitionFactorRBFMatrix(Partition p)
{
  MatFactorInfo factinfo;
  IS            perm,cperm;
  Mat           A;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_FactorRBFMatrix,0,0,0,0));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &perm, &cperm));
  PetscCall(ISDestroy(&cperm));
  /* initialize factinfo */
  PetscCall(MatFactorInfoInitialize(&factinfo));
  /* factor */
  PetscCall(MatCholeskyFactor(A,perm,&factinfo));
  PetscCall(ISDestroy(&perm));
  PetscCall(PetscLogEventEnd(LAGRBF_FactorRBFMatrix,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B)
{
  Mat       RHS;
  PetscInt  f,np,npoints;
  PetscInt  *idx;
  PetscReal *LA_rhs;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_MatSetRHS_Fac,0,0,0,0));
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
      //LA_rhs[f*npoints + np] = fval[ f*npoints + idx[np] ]; // set the RHS values
    }
  }
  PetscCall(MatDenseRestoreArray(RHS,&LA_rhs));
  *B = RHS;
  PetscCall(PetscLogEventEnd(LAGRBF_MatSetRHS_Fac,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#if 0
static PetscErrorCode PartitionMatSetRHS_Columnwise(Partition p, PetscInt nfields, PetscReal fval[], Mat *B)
{
  Mat       RHS = NULL;
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
#endif
PetscErrorCode PartitionSolveLocalCoefficientsFactor(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[])
{
  Mat      A,B,X;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SolveLocalCoefficients_Fac,0,0,0,0));
  PetscCall(PartitionGetCoefficientsMat(p,&X));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionMatSetRHS(p,nfields,fval,&B));
  /* solve the linear system A*X = B */
  PetscCall(MatMatSolve(A,B,X));
  /* Clean-up */
  PetscCall(MatDestroy(&B));
  PetscCall(PetscLogEventEnd(LAGRBF_SolveLocalCoefficients_Fac,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetUpRBFMatrix_Factor(LagRBFCtx ctx, Partition p, PetscReal *coords)
{
  PartitionOfUnity pu;
  KernelType       ktype;
  PetscInt         dim;
  PetscReal        epsilon;
  PetscReal        (*kernel_fn)(PetscReal,PetscReal);
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetEpsilon(pu,&epsilon));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(LagRBFGetKernelType(ctx,&ktype));
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  /* Create the RBF matrix, fill it and factor it */
  PetscCall(PartitionCreateRBFMatrix(p));
  PetscCall(PartitionAssembleRBFMatrix(p,dim,epsilon,coords,kernel_fn));
  PetscCall(PartitionFactorRBFMatrix(p));

  PetscFunctionReturn(PETSC_SUCCESS);
}