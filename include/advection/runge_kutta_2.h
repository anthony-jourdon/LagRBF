#ifndef __LagRBF_runge_kutta_2_h
#define __LagRBF_runge_kutta_2_h

#include "advection/advection_ctx.h"

PetscErrorCode Advection_RK2Create(AdvectionCtx adv_ctx);
PetscErrorCode AdvectionExplicit_RK2(LagRBFCtx ctx, DM swi, DM swd);

#endif