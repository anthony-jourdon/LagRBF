#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "advection/advection_ctx.h"
#include "cell_info.h"
#include "kernels/kernels_rbf.h"

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx)
{
  LagRBFCtx p;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&p));
  p->pu           = NULL;
  p->adv_ctx      = NULL; 
  p->mtype        = GEODYNAMICS; // Default model type
  p->partitions   = NULL;
  p->cellinfo     = NULL;
  p->kernel_type  = KERNEL_LINEAR;
  p->swd          = NULL;
  p->swi          = NULL;
  p->n_ifields    = 0;
  p->n_pfields    = 0;
  p->step         = 0;
  p->time         = 0.0;
  p->restart      = PETSC_FALSE;
  p->advection    = PETSC_FALSE;
  p->solve_method = A_FACTOR;
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
  if (p->cellinfo)   { PetscCall(CellInfoDestroy(p->pu, &p->cellinfo)); }
  if (p->pu)         { PetscCall(PartitionOfUnityDestroy(&p->pu)); }
  if (p->adv_ctx)    { PetscCall(LagRBF_AdvectionCtxDestroy(p)); }
  if (p->swd)        { PetscCall(DMDestroy(&p->swd)); }
  if (p->swi)        { PetscCall(DMDestroy(&p->swi)); }
  PetscCall(PetscFree(p));
  *ctx = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_PartitionOfUnityCreate(LagRBFCtx ctx)
{
  PartitionOfUnity pu = NULL;
  PetscFunctionBeginUser;
  PetscCall(PartitionOfUnityCreate(&pu));
  PetscCall(PUAttachPartitionFunctions(pu, ctx->solve_method));
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

PetscErrorCode LagRBF_CellInfoCreate(LagRBFCtx ctx)
{
  PartitionOfUnity pu = NULL;
  CellInfo        *cellinfo;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(CellInfoCreate(pu,&cellinfo));
  ctx->cellinfo = cellinfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetFromOptions(LagRBFCtx ctx)
{
  PetscInt  scheme,kernel_type,solve_method;
  PetscBool found;
  char option_name[PETSC_MAX_PATH_LEN],velocity_name[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-n_interpolation_fields");
  PetscCall(PetscOptionsGetInt(NULL, NULL, option_name, &ctx->n_ifields, NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-output_path",ctx->output_dir,PETSC_MAX_PATH_LEN-1,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-kernel_type",&kernel_type,&found));
  if (found) { PetscCall(LagRBFSetKernelType(ctx,(KernelType)kernel_type)); }
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-solve_method",&solve_method,&found));
  if (found) { PetscCall(LagRBFSetSolveMethod(ctx,(ASolveMethod)solve_method)); }
  
  /* Check if advection is required, if not return */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-advection",&ctx->advection,NULL));
  if (!ctx->advection) { PetscFunctionReturn(PETSC_SUCCESS); }
  /* Advection specific options */
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1,"-advection_scheme");
  scheme = 3; // Default to DOPRI
  PetscCall(PetscOptionsGetInt(NULL, NULL, option_name,&scheme, NULL));
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1,"-velocity_name");
  PetscSNPrintf(velocity_name, PETSC_MAX_PATH_LEN-1, "velocity");
  PetscCall(PetscOptionsGetString(NULL,NULL,option_name,velocity_name,PETSC_MAX_PATH_LEN-1,NULL));
  PetscCall(LagRBF_AdvectionCtxCreate(ctx,(AdvectionScheme)scheme,velocity_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_AdvectSwarm(LagRBFCtx ctx, DM swi, DM swd)
{
  AdvectionCtx adv_ctx=NULL;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  /* Advect */
  adv_ctx->advection(ctx,swi,swd);
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
  PetscCall(AdvectionSetTimeStep(ctx->adv_ctx, dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetTime(LagRBFCtx ctx, PetscReal time)
{
  PetscFunctionBeginUser;
  ctx->time = time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetKernelType(LagRBFCtx ctx, KernelType type)
{
  PetscFunctionBeginUser;
  ctx->kernel_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFSetSolveMethod(LagRBFCtx ctx, ASolveMethod method)
{
  PetscFunctionBeginUser;
  ctx->solve_method = method;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetSolveMethod(LagRBFCtx ctx, ASolveMethod *method)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx context is NULL.");
  if (method) { *method = ctx->solve_method; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetKernelType(LagRBFCtx ctx, KernelType *type)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx context is NULL.");
  if (type) { *type = ctx->kernel_type; }
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

PetscErrorCode LagRBFGetCellInfo(LagRBFCtx ctx, CellInfo **cellinfo)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->cellinfo, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "CellInfo context is NULL. Call LagRBF_CellInfoCreate() first.");
  if (cellinfo) { *cellinfo = ctx->cellinfo; }
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
  PetscReal time_step;
  PetscFunctionBeginUser;
  if (dt) { 
    PetscCall(AdvectionGetTimeStep(ctx->adv_ctx, &time_step)); 
    *dt = time_step; 
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetTime(LagRBFCtx ctx, PetscReal *time)
{
  PetscFunctionBeginUser;
  if (time) { *time = ctx->time; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetAdvectionCtx(LagRBFCtx ctx, AdvectionCtx *adv_ctx)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->adv_ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL. Call AdvectionCtxCreate() first.");
  if (adv_ctx) { *adv_ctx = ctx->adv_ctx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetSwarmData(LagRBFCtx ctx, DM *swd)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->swd, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Swarm for data points is NULL. Call CreateSwarmData() first.");
  if (swd) { *swd = ctx->swd; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFGetSwarmInterp(LagRBFCtx ctx, DM *swi)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx->swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Swarm for interpolation points is NULL. Call CreateSwarmInterp() first.");
  if (swi) { *swi = ctx->swi; }
  PetscFunctionReturn(PETSC_SUCCESS);
}