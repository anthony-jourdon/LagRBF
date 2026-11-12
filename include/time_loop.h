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
  file:    time_loop.h

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

#ifndef __LagRBF_time_loop_h
#define __LagRBF_time_loop_h

#include "lagrangian_rbf.h"

PetscErrorCode InternalForwardAdvectionLoop(LagRBFCtx ctx, DM swarm_data, DM swarm_interp, PetscReal time, PetscReal time_next);
PetscErrorCode ForwardTimeLoop(LagRBFCtx ctx, DM swarm_data, DM swarm_interp, PetscInt nsteps, PetscInt steps[], PetscReal time_steps[], PetscInt restart_idx);
PetscErrorCode ForwardAnalyticalAdvection(LagRBFCtx ctx, DM sw, PetscReal time, PetscReal time_next);

#endif