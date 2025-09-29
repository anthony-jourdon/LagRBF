#ifndef __softname_swarm_utils_h
#define __softname_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode CopyContiguousBlockData_Generic(void *dest, void *source, PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt start_block, size_t typesize);
#define CopyContiguousBlockData(dest, source, npoints, nblocks, bs, start_block) \
  CopyContiguousBlockData_Generic(dest, source, npoints, nblocks, bs, start_block, sizeof(*(dest)))

PetscErrorCode CopyContiguousBlockData_Real(PetscReal *dest, PetscReal *source, PetscInt  npoints, PetscInt  nblocks, PetscInt  bs, PetscInt  start_block);
PetscErrorCode CopyArbitraryBlockData_Real(PetscReal dest[], PetscReal source[], PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt blocks_to_copy[]);
PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode);

#endif