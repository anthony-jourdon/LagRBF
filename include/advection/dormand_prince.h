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
  file:    dormand_prince.h

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