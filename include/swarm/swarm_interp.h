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
  file:    swarm_interp.h

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

#ifndef __LagRBF_swarm_interp_h
#define __LagRBF_swarm_interp_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmInterpCreateAdvectionFields(LagRBFCtx ctx, DM swarm);
PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw);
//PetscErrorCode SwarmInterpSetWeights_Opt(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetCoordinatesFromOptions(PartitionOfUnity pu, DM sw, InsertMode mode);
PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx);
PetscErrorCode SwarmInterpAddMarkersFromOptions(LagRBFCtx ctx, DM sw, InsertMode mode);
PetscErrorCode SwarmInterpLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmInterpLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmInterpLoadFromFile(LagRBFCtx ctx, DM sw, const char *dirname, PetscBool load_pidx);

#if defined (__AVX__)
PetscErrorCode SwarmInterpSetWeights_AVX256(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetWeights_AVX256_PointVectorized(LagRBFCtx ctx, DM sw);
#endif
#if defined (__AVX512F__)
PetscErrorCode SwarmInterpSetWeights_AVX512(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetWeights_AVX512_PointVectorized(LagRBFCtx ctx, DM sw);
#endif

#endif