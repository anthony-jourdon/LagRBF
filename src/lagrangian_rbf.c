#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx)
{
  LagRBFCtx p;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&p));
  p->pu         = NULL;
  p->mtype      = GEODYNAMICS; // Default model type
  p->partitions = NULL;
  p->int_order  = 1;
  p->n_ifields  = 0;
  p->n_pfields  = 0;
  p->step       = 0;
  p->time       = 0.0;
  p->dt         = 0.0;
  p->restart    = PETSC_FALSE;
  PetscCall(PetscSNPrintf(p->output_dir, PETSC_MAX_PATH_LEN-1, "output"));

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
  PartitionOfUnity pu = NULL;
  PetscFunctionBeginUser;
  PetscCall(PartitionOfUnityCreate(&pu));
  ctx->pu = pu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_PartitionCreate(LagRBFCtx ctx)
{
  PartitionOfUnity pu = NULL;
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
  PetscCall(PetscOptionsGetString(NULL,NULL,"-output_path",ctx->output_dir,PETSC_MAX_PATH_LEN-1,NULL));
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-advection_scheme");
  PetscCall(PetscOptionsGetInt(NULL, NULL, option_name, &ctx->int_order, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetStep(LagRBFCtx ctx, PetscInt step)
{
  PetscFunctionBeginUser;
  ctx->step = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetTimeStep(LagRBFCtx ctx, PetscReal dt)
{
  PetscFunctionBeginUser;
  ctx->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetTime(LagRBFCtx ctx, PetscReal time)
{
  PetscFunctionBeginUser;
  ctx->time = time;
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

PetscErrorCode LagRBFGetStep(LagRBFCtx ctx, PetscInt *step)
{
  PetscFunctionBeginUser;
  if (step) { *step = ctx->step; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetTimeStep(LagRBFCtx ctx, PetscReal *dt)
{
  PetscFunctionBeginUser;
  if (dt) { *dt = ctx->dt; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetTime(LagRBFCtx ctx, PetscReal *time)
{
  PetscFunctionBeginUser;
  if (time) { *time = ctx->time; }
  PetscFunctionReturn(PETSC_SUCCESS);
}