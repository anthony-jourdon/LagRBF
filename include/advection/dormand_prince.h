#ifndef __LagRBF_dormand_prince_4_h
#define __LagRBF_dormand_prince_4_h

#include <petsc.h>

typedef struct {
  PetscReal dt_fac;
  PetscReal dt_fac_min;
  PetscReal dt_fac_max;
  PetscReal a_tol,r_tol;
  PetscReal error;
} DoPriCtx;

PetscErrorCode AdvectionEvaluateError_DOPRI(AdvectionCtx ctx, DM swi);
PetscErrorCode AdvectionExplicit_DOPRI(LagRBFCtx ctx, DM swi, DM swd);
PetscErrorCode Advection_DOPRICreate(AdvectionCtx adv_ctx);

#endif