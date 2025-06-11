#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx)
{
  LagRBFCtx p;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&p));
  p->pu         = NULL;
  p->partitions = NULL;
  p->n_ifields  = 0;
  p->n_pfields  = 0;
  p->restart    = PETSC_FALSE;

  *ctx = p;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFCtxDestroy(LagRBFCtx *ctx)
{
  LagRBFCtx p = *ctx;
  PetscFunctionBeginUser;

  if (!p)            { PetscFunctionReturn(PETSC_SUCCESS); }
  if (p->partitions) { PetscCall(PartitionDestroy(p->pu, &p->partitions)); }
  if (p->pu)         { PetscCall(PartitionOfUnityDestroy(&p->pu)); }
  PetscCall(PetscFree(p));
  *ctx = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_PartitionOfUnityCreate(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  PetscFunctionBeginUser;
  PetscCall(PartitionOfUnityCreate(&pu));
  ctx->pu = pu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_PartitionCreate(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PartitionCreate(pu,&partitions));
  ctx->partitions = partitions;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetFromOptions(LagRBFCtx ctx)
{
  char option_name[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-n_interpolation_fields");
  PetscCall(PetscOptionsGetInt(NULL, NULL, option_name, &ctx->n_ifields, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetPartitionOfUnity(LagRBFCtx ctx, PartitionOfUnity *pu)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->pu, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Partition of Unity context is NULL. Call LagRBF_PartitionOfUnityCreate() first.");
  if (pu) { *pu = ctx->pu; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetPartitions(LagRBFCtx ctx, Partition **partitions)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->partitions, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Partitions context is NULL. Call LagRBF_PartitionCreate() first.");
  if (partitions) { *partitions = ctx->partitions; }
  PetscFunctionReturn(PETSC_SUCCESS);
}