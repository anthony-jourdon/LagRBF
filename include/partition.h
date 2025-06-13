#ifndef __softname_partition_h
#define __softname_partition_h

#include <petsc.h>
#include "partition_of_unity.h"
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
  PetscReal epsilon; // Epsilon factor for the RBF kernel
};

typedef struct _p_Partition *Partition;

PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition);
PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition);
PetscErrorCode PartitionSetUp_All(Partition *partitions, PetscInt npart, DM swarm);
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