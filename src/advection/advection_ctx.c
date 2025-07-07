#include <petsc.h>
#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "advection/runge_kutta_1.h"
#include "advection/runge_kutta_4.h"

const char *AdvectionSchemeNames[] = {
  "RK1", // First order Runge-Kutta
  "RK2", // Second order Runge-Kutta
  "RK4", // Fourth order Runge-Kutta
  "DOPRI", // Dormand-Prince 5(4) method
  NULL
};

PetscErrorCode (*create_scheme[])(AdvectionCtx) = {
  Advection_RK1Create,
  NULL, // No RK2 implementation
  Advection_RK4Create,
  NULL, // No DOPRI implementation
  0
};

PetscErrorCode AdvectionCtxCreate(AdvectionCtx *ctx)
{
  AdvectionCtx adv;
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&adv));
  adv->user = NULL;
  *ctx = adv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionCtxDestroy(AdvectionCtx *ctx)
{
  AdvectionCtx adv = *ctx;
  PetscFunctionBeginUser;

  if (!adv) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (adv->destroy) { PetscCall(adv->destroy(adv)); }
  PetscCall(PetscFree(adv));
  *ctx = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionCtxSetScheme(AdvectionCtx ctx, AdvectionScheme scheme)
{
  int        k=0;
  PetscBool  found;
  const char *name;
  PetscErrorCode (*create)(AdvectionCtx) = NULL;
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL");
  /* Assign the scheme */
  ctx->scheme = scheme;
  name = AdvectionSchemeNames[k];
  while (name != NULL) {
    /* Check if name matches the type given */
    PetscCall(PetscStrcmp(name, AdvectionSchemeNames[ (int)scheme ], &found));
    if (found) {
      create = create_scheme[k];
      break;
    }
    k++;
    /* Assign name to the next entry of the array and restart */
    name = AdvectionSchemeNames[k];
  }
  PetscCheck(create, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Advection scheme %s Create function not found", AdvectionSchemeNames[(int)scheme]);
  PetscCall(create(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionSetTimeStep(AdvectionCtx ctx, PetscReal dt)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL");
  ctx->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionGetTimeStep(AdvectionCtx ctx, PetscReal *dt)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL");
  if (dt) { *dt = ctx->dt; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionSetVelocityName(AdvectionCtx ctx, const char *name)
{
  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(ctx->velocity_name,PETSC_MAX_PATH_LEN-1,"%s",name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionGetVelocityName(AdvectionCtx ctx, const char **velocity_name)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL");
  if (velocity_name) { *velocity_name = ctx->velocity_name; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionGetIntegrationOrder(AdvectionCtx ctx, PetscInt *order)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Advection context is NULL");
  if (order) { *order = ctx->order; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_AdvectionCtxCreate(LagRBFCtx ctx, AdvectionScheme scheme, const char *velocity_name)
{
  AdvectionCtx adv = NULL;
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx context is NULL");
  
  /* Use this function to set everything */
  PetscCall(AdvectionCtxCreate(&adv));
  PetscCall(AdvectionCtxSetScheme(adv,scheme));
  PetscCall(AdvectionSetVelocityName(adv,velocity_name));
  ctx->adv_ctx = adv;
  PetscFunctionReturn(PETSC_SUCCESS);
}
