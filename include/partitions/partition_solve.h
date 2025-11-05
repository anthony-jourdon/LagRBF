#ifndef __softname_partition_solve_h
#define __softname_partition_solve_h

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition.h"

PetscErrorCode PartitionCreateRBFMatrix(Partition p);
PetscErrorCode PartitionAssembleRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionSetUpRBFMatrix_Factor(Partition p, PetscInt dim, PetscReal coords[], PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionSetUpRBFMatrix_KSP(Partition p, PetscInt dim, PetscReal coords[], PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B);
PetscErrorCode PartitionSolveInterpolationCoefficients(PartitionOfUnity pu, Partition *partitions, DM swarm, const char *fieldname);

#endif