#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "time_loop.h"
#include "interface/interface_utils.h"
#include "interface/interface_adv.h"
#include "advection/advection.h"

PetscErrorCode lagrbf_advection_function(
  void *ctx,
  PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), 
  void *fn_ctx
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  AdvectionCtx   adv_ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(user,&adv_ctx));
  ierr = AdvectionSetVelocityFunction(adv_ctx, velocity_fn, fn_ctx);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "AdvectionSetVelocityFunction");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_advection_function_batch(
  void *ctx,
  PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), 
  void *fn_ctx
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  AdvectionCtx   adv_ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(user,&adv_ctx));
  ierr = AdvectionSetVelocityFunctionBatch(adv_ctx, velocity_fn_batch, fn_ctx);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "AdvectionSetVelocityFunctionBatch");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_avect_analytical(void *ctx, PetscReal time, PetscReal time_next)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  PetscCheck(user,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx is NULL");
  PetscCheck(user->swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Interpolation swarm (ctx->swi) is NULL");
  ierr = ForwardAnalyticalAdvection(user, user->swi, time, time_next);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "ForwardAnalyticalAdvection");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_advect_interpolate(void *ctx, PetscReal time, PetscReal time_next)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCheck(user,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx is NULL");
  PetscCheck(user->swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Interpolation swarm (ctx->swi) is NULL");
  PetscCheck(user->swd, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Data swarm (ctx->swd) is NULL");

  ierr = InternalForwardAdvectionLoop(user,user->swd,user->swi,time,time_next);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "InternalForwardAdvectionLoop");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}