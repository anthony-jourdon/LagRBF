#ifndef __softname_partition_h
#define __softname_partition_h

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
  void      *data;
};

PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition);
PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition);
PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi);
PetscErrorCode PartitionFillDataPointsIdx(Partition p, DM swarm);
/* Getters and setters */
PetscErrorCode PartitionGetMap(Partition p, PetscInt *ncells, PetscInt **map);
PetscErrorCode PartitionGetDataPointsIdx(Partition p, PetscInt *ndata, PetscInt **data_idx);
PetscErrorCode PartitionGetCoefficientsMat(Partition p, Mat *X);
PetscErrorCode PartitionGetRBFMatrix(Partition p, Mat *A);
PetscErrorCode PartitionGetEpsilon(Partition p, PetscReal *epsilon);
PetscErrorCode PartitionSetEpsilon(Partition p, PetscReal epsilon);
PetscErrorCode PartitionsSetIndex(PartitionOfUnity pu, Partition *partitions);
PetscErrorCode PartitionGetIndex(Partition p, PetscInt *idx);
#endif