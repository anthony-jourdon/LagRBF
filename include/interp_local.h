#ifndef __softname_interp_local_h
#define __softname_interp_local_h

#include <petsc.h>
#include "partition.h"

PetscErrorCode PartitionAssembleRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal), Mat *A);
PetscErrorCode PartitionFactorizeRBFMatrix(Partition p,PetscInt dim,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal),Mat *A);

#endif