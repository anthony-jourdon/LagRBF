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
  file:    partition_ksp.c

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
#include "partitions/partition.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_MatSetRHS_KSP;
PetscLogEvent LAGRBF_SolveLocalCoefficients_KSP;

typedef struct {
  KSP ksp;
  Vec RHS;
} PSolve_KSPData;

static PetscErrorCode PartitionKSPCreateCtx(PSolve_KSPData **data)
{
  PSolve_KSPData *ctx;
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&ctx));
  *data = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionKSPDestroyCtx(Partition p)
{
  PSolve_KSPData *data_ctx = (PSolve_KSPData*)p->data;
  PetscFunctionBeginUser;
  if (!data_ctx)     { PetscFunctionReturn(PETSC_SUCCESS); }
  if (data_ctx->RHS) { PetscCall(VecDestroy(&data_ctx->RHS)); }
  if (data_ctx->ksp) { PetscCall(KSPDestroy(&data_ctx->ksp)); }
  PetscCall(PetscFree(data_ctx));
  p->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionKSPCreate(Partition p, KSP *k)
{
  Mat      A;
  KSP      ksp;
  MPI_Comm comm;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(KSPCreate(comm,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"partition_"));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  *k = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionRHSVecCreate(Partition p, Vec *RHS)
{
  Vec       rhs;
  PetscInt  npoints;
  PetscFunctionBeginUser;
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
  PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"partition_rhs",1,npoints,PETSC_DETERMINE,&rhs));
  *RHS = rhs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionSetFieldRHS(Partition p, PetscInt nfields, PetscInt field_idx, PetscReal fval[], Vec RHS)
{
  PetscInt  np,npoints;
  PetscInt  *idx;
  PetscReal *buffer,*LA_RHS;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_MatSetRHS_KSP,0,0,0,0));
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&idx));
  //for (np=0; np<npoints; np++) {
  //  PetscCall(VecSetValue(RHS,np,fval[idx[np]*nfields + field_idx],INSERT_VALUES));
  //}
  #if 1
  PetscCall(PetscMalloc1(npoints,&buffer));
  for (np=0; np<npoints; np++) {
    buffer[np] = fval[ idx[np]*nfields + field_idx ]; // set the RHS values for the field
  }
  PetscCall(VecGetArray(RHS,&LA_RHS));
  PetscCall(PetscArraycpy(LA_RHS,buffer,npoints));
  PetscCall(VecRestoreArray(RHS,&LA_RHS));
  PetscCall(PetscFree(buffer));
  #endif
  PetscCall(PetscLogEventEnd(LAGRBF_MatSetRHS_KSP,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSolveLocalCoefficientsKSP(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[])
{
  PSolve_KSPData *data_ctx = (PSolve_KSPData*)p->data;
  Mat            A,X;
  Vec            Xf,b;
  PetscInt       f;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SolveLocalCoefficients_KSP,0,0,0,0));
  PetscCall(PartitionGetCoefficientsMat(p,&X));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  b = data_ctx->RHS;
  for (f=0; f<nfields; f++) {
    PetscCall(MatDenseGetColumnVec(X,f,&Xf));
    PetscCall(PartitionSetFieldRHS(p,nfields,f,fval,b));
    PetscCall(KSPSolve(data_ctx->ksp,b,Xf));
    PetscCall(MatDenseRestoreColumnVec(X,f,&Xf));
  }
  PetscCall(PetscLogEventEnd(LAGRBF_SolveLocalCoefficients_KSP,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSolveLocalCoefficientsKSP_ASMB(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[])
{
  PSolve_KSPData *data_ctx = (PSolve_KSPData*)p->data;
  PetscFunctionBeginUser;
  PetscCall(PartitionSolveLocalCoefficientsKSP(ctx,p,nfields,fval));
  /* Free memory after solve */
  PetscCall(KSPReset(data_ctx->ksp));
  PetscCall(MatDestroy(&p->A));
  p->A = NULL; // Mark the RBF matrix as destroyed in the partition structure
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetUpRBFMatrix_KSP(LagRBFCtx ctx, Partition p, PetscReal *coords)
{
  PartitionOfUnity pu;
  KernelType       ktype;
  PSolve_KSPData   *data_ctx = NULL;
  PetscInt         dim;
  PetscReal        epsilon;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetShapeFactor(pu,&epsilon));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(LagRBFGetKernelType(ctx,&ktype));
  /* Method-specific context */
  PetscCall(PartitionKSPCreateCtx(&data_ctx));
  /* Create and assemble RBF matrix */
  PetscCall(PartitionCreateRBFMatrix(p));
  PetscCall(PartitionAssembleRBFMatrix(p,dim,epsilon,coords,ktype));
  /* Create KSP solver context */
  PetscCall(PartitionKSPCreate(p,&data_ctx->ksp));
  /* Create RHS vector */
  PetscCall(PartitionRHSVecCreate(p,&data_ctx->RHS));
  /* Attach method-specific context */
  p->data = (void*)data_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}