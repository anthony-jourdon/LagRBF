/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    advection.h

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#ifndef __LagRBF_advection_h
#define __LagRBF_advection_h

#include "advection/advection_ctx.h"

PetscErrorCode AdvectionSetInitialValues(AdvectionCtx ctx, DM swi);
PetscErrorCode AdvectionExplicit(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);
PetscErrorCode AdvectionExplicit_Vec(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim);
PetscErrorCode AdvectionComputeTimestep_CFL(LagRBFCtx ctx, DM swarm, PetscReal *_dt);

PetscErrorCode AdvectionProjectOntoSphereBoundary_Pointwise(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax);
PetscErrorCode AdvectionProjectOntoSphereBoundary(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax, PetscInt npoints);

PetscErrorCode AdvectionCorrectSphereBoundaryPosition_Pointwise(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt dim, PetscReal Rmin2, PetscReal Rmax2);
PetscErrorCode AdvectionCorrectSphereBoundaryPosition(PetscReal *coords, PetscReal *x0, PetscReal *velocity, PetscReal dt, PetscInt npoints, PetscInt dim, PetscReal Rmin, PetscReal Rmax);
#endif