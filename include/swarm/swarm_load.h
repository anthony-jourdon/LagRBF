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
  file:    swarm_load.h

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

#ifndef __LagRBF_swarm_load_h
#define __LagRBF_swarm_load_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmLoadVecFromFile(DM sw, const char *field_name, const char *filename, MPI_Comm comm, PetscBool resize);
PetscErrorCode SwarmLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmLoadFromFiles(LagRBFCtx ctx, DM sw, const char *dirname, const char **fields_name, PetscInt nfields);

#endif