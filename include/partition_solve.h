#ifndef __softname_partition_solve_h
#define __softname_partition_solve_h

#include <petsc.h>
#include "partition.h"

PetscErrorCode PartitionSetUpRBFMatrix(Partition p, PetscInt dim, PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B);
PetscErrorCode PartitionSolveInterpolationCoefficients(PartitionOfUnity pu, Partition *partitions, DM swarm);

#endif