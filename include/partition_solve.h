#ifndef __softname_partition_solve_h
#define __softname_partition_solve_h

#include <petsc.h>
#include "partition.h"

PetscErrorCode PartitionAssembleRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal), Mat *A);
PetscErrorCode PartitionFactorRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal),Mat *A);
PetscErrorCode PartitionSetUpRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionSetVecRHS(Partition p, PetscReal fvals[], Vec *B);

#endif