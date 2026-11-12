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
  file:    swarm_utils.h

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

#ifndef __LagRBF_swarm_utils_h
#define __LagRBF_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmFieldAssignUniqueIndex(DM sw, const char *field_name, PetscInt start_idx, PetscInt end_idx);

PetscErrorCode CopyContiguousBlockData_Generic(void *dest, void *source, PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt start_block, size_t typesize);
#define CopyContiguousBlockData(dest, source, npoints, nblocks, bs, start_block) \
  CopyContiguousBlockData_Generic(dest, source, npoints, nblocks, bs, start_block, sizeof(*(dest)))

PetscErrorCode CopyContiguousBlockData_Real(PetscReal *dest, PetscReal *source, PetscInt  npoints, PetscInt  nblocks, PetscInt  bs, PetscInt  start_block);
PetscErrorCode CopyArbitraryBlockData_Real(PetscReal dest[], PetscReal source[], PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt blocks_to_copy[]);
PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscReal halton_sequence(PetscInt index, PetscInt base);
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode, PetscBool halton);
PetscErrorCode DMSwarmSetPointsCoordinates_FromOptions(DM sw, InsertMode mode);
PetscErrorCode DMSwarmRemoveNPointsFromIndex(DM swarm, PetscInt nindices, PetscInt *indices);
PetscErrorCode DMSwarmPICSetCellId(DM sw);
PetscErrorCode DMSwarmMigrate_Push_DuplicatePoints(DM swarm, PetscInt point_start, PetscInt point_end);
PetscErrorCode DMSwarmAssignRankFromDMDA(DM swarm, DM da);
PetscErrorCode DMSwarmAssignMigrateBasic(LagRBFCtx ctx, DM swarm);
#endif