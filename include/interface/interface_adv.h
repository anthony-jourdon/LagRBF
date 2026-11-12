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
  file:    interface_adv.h

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

#ifndef __LagRBF_interface_adv_h
#define __LagRBF_interface_adv_h

#include "lagrangian_rbf.h"

LAGRBF_EXPORT PetscErrorCode lagrbf_advection_function(
  void *ctx,
  PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), 
  void *fn_ctx
);

LAGRBF_EXPORT PetscErrorCode lagrbf_advection_function_batch(
  void *ctx,
  PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), 
  void *fn_ctx
);

LAGRBF_EXPORT PetscErrorCode lagrbf_avect_analytical(
  void *ctx, 
  PetscReal time, 
  PetscReal time_next
);

LAGRBF_EXPORT PetscErrorCode lagrbf_advect_interpolate(
  void *ctx, 
  PetscReal time, 
  PetscReal time_next
);

#endif