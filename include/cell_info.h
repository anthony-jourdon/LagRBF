#ifndef __softname_cell_info_h
#define __softname_cell_info_h

#include <petsc.h>
#include "partition_of_unity.h"

struct _p_CellInfo {
  PetscInt  idx;         // Cell index
  PetscInt  nneigh;      // Number of neighbours -> 9 in 2D, 27 in 3D
  PetscInt  *neighbours; // Neighbour cells indices
  PetscInt  ndata;       // Number of data points connected to the cell (include neighbours)
  PetscInt  *data_idx;   // Data points indices connected to the cell (include neighbours)
};
typedef struct _p_CellInfo *CellInfo;

PetscErrorCode CellInfoCreate(PartitionOfUnity pu, CellInfo **cellinfo);
PetscErrorCode CellInfoDestroy(PartitionOfUnity pu, CellInfo **cellinfo);
PetscErrorCode CellInfoCreateNeighbourElements(PartitionOfUnity pu, CellInfo *ci);
PetscErrorCode CellInfoFillDataIdx(CellInfo ci, DM swarm);
PetscErrorCode CellInfoSetUp(PartitionOfUnity pu, CellInfo *ci, DM swarm);
/* Getters */
PetscErrorCode CellInfoGetIndex(CellInfo ci, PetscInt *idx);
PetscErrorCode CellInfoGetNumberOfNeighbours(CellInfo ci, PetscInt *nneigh);
PetscErrorCode CellInfoGetNeighbours(CellInfo ci, PetscInt *nneigh, PetscInt **neighb);
PetscErrorCode CellInfoGetDataPointsIdx(CellInfo ci, PetscInt *ndata, PetscInt **data_idx);
#endif