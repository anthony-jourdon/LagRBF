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
  file:    cell_info.h

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

#ifndef __LagRBF_cell_info_h
#define __LagRBF_cell_info_h

#include <petsc.h>
#include "partition_of_unity.h"

struct _p_CellInfo {
  PetscInt  idx;         // Cell index
  PetscInt  nneigh;      // Number of neighbours -> 9 in 2D, 27 in 3D
  PetscInt  *neighbours; // Neighbour cells indices
  PetscInt  ndata;       // Number of data points connected to the cell (include neighbours)
  PetscInt  *data_idx;   // Data points indices connected to the cell (include neighbours)
  PetscInt  *node_stencil; // Node stencil indices for the cell
};

PetscErrorCode CellInfoCreate(PartitionOfUnity pu, CellInfo **cellinfo);
PetscErrorCode CellInfoDestroy(PartitionOfUnity pu, CellInfo **cellinfo);
PetscErrorCode CellInfoCreateNeighbourElements(PartitionOfUnity pu, CellInfo *ci);
PetscErrorCode CellInfoFillDataIdx(CellInfo ci, DM swarm);
PetscErrorCode CellInfoSetUp(PartitionOfUnity pu, CellInfo *ci, DM swarm);
PetscErrorCode CellInfoCreateNodeStencils(PartitionOfUnity pu, CellInfo *ci);
PetscErrorCode CellInfoReset(PartitionOfUnity pu, CellInfo *ci);
/* Getters */
PetscErrorCode CellInfoGetIndex(CellInfo ci, PetscInt *idx);
PetscErrorCode CellInfoGetNumberOfNeighbours(CellInfo ci, PetscInt *nneigh);
PetscErrorCode CellInfoGetNeighbours(CellInfo ci, PetscInt *nneigh, PetscInt **neighb);
PetscErrorCode CellInfoGetDataPointsIdx(CellInfo ci, PetscInt *ndata, PetscInt **data_idx);
PetscErrorCode DMDAGetElementLogicalIndex(DM da, PetscInt eidx, PetscInt *ix, PetscInt *iy, PetscInt *iz);
PetscErrorCode CellInfoGetStencilNodes(CellInfo ci, PetscInt **stencil);
#endif