#ifndef __softname_advection_h
#define __softname_advection_h

#include "advection/advection_ctx.h"

PetscErrorCode AdvectionSetInitialValues(AdvectionCtx ctx, DM swi);
PetscErrorCode AdvectionExplicit(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);
PetscErrorCode AdvectionExplicit_Vec(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);

#endif