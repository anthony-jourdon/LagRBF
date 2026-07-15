#ifndef __LagRBF_runge_kutta_1_h
#define __LagRBF_runge_kutta_1_h

#include "advection/advection_ctx.h"
PetscErrorCode AdvectionExplicit_RK1(LagRBFCtx ctx, DM swi, DM swd);
PetscErrorCode Advection_RK1Create(AdvectionCtx adv_ctx);
#endif