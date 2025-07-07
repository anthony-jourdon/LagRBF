#ifndef __softname_runge_kutta_4_h
#define __softname_runge_kutta_4_h

#include "advection/advection_ctx.h"

PetscErrorCode Advection_RK4Create(AdvectionCtx adv_ctx);
PetscErrorCode AdvectionExplicit_RK4(LagRBFCtx ctx, DM swi, DM swd);

#endif