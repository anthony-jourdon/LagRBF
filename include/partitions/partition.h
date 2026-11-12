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
  file:    partition.h

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

#ifndef __LagRBF_partition_h
#define __LagRBF_partition_h

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"

//struct 

/* 
Partition of Unity Method
a partition is a node of the DMDA 
*/
struct _p_Partition {
  Mat       A; // RBF matrix for the partition
  Mat       X; // Matrix to hold the coefficients for the partition interpolation
  PetscInt  ncells; // Number of cells connected to the partition
  PetscInt  *map; // Inverse map from partition to cells
  PetscInt  ndata; // Number of data points in the partition
  PetscInt  idx; // Index of the partition
  PetscInt  *data_idx; // Indices of data points in the partition
  PetscBool setup;
  PetscBool solved;
  void      *data;
};

PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition);
PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition);
PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi);
PetscErrorCode PartitionFillDataPointsIdx(Partition p, DM swarm);
PetscErrorCode LagRBF_PartitionsReset(LagRBFCtx ctx);
/* Getters and setters */
PetscErrorCode PartitionGetMap(Partition p, PetscInt *ncells, PetscInt **map);
PetscErrorCode PartitionGetDataPointsIdx(Partition p, PetscInt *ndata, PetscInt **data_idx);
PetscErrorCode PartitionGetCoefficientsMat(Partition p, Mat *X);
PetscErrorCode PartitionGetRBFMatrix(Partition p, Mat *A);
PetscErrorCode PartitionGetShapeFactor(Partition p, PetscReal *epsilon);
PetscErrorCode PartitionSetShapeFactor(Partition p, PetscReal epsilon);
PetscErrorCode PartitionsSetIndex(PartitionOfUnity pu, Partition *partitions);
PetscErrorCode PartitionGetIndex(Partition p, PetscInt *idx);
#endif