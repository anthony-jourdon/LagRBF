#ifndef __softname_advection_h
#define __softname_advection_h

#include "advection/advection_ctx.h"

PetscErrorCode AdvectionSetInitialValues(AdvectionCtx ctx, DM swi);
PetscErrorCode AdvectionExplicit(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);
PetscErrorCode AdvectionExplicit_Vec(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);
PetscErrorCode AdvectionComputeTimestep_CFL(LagRBFCtx ctx, DM swarm);

PetscErrorCode AdvectionProjectOntoSphereBoundary_Pointwise(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax);
PetscErrorCode AdvectionProjectOntoSphereBoundary(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax, PetscInt npoints);

PetscErrorCode AdvectionCorrectSphereBoundaryPosition_Pointwise(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt dim, PetscReal Rmin2, PetscReal Rmax2);
PetscErrorCode AdvectionCorrectSphereBoundaryPosition(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim, PetscReal Rmin, PetscReal Rmax);
#endif