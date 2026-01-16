#ifndef __softname_swarm_utils_h
#define __softname_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmFieldAssignUniqueIndex(DM sw, const char *field_name, PetscInt start_idx, PetscInt end_idx);

PetscErrorCode CopyContiguousBlockData_Generic(void *dest, void *source, PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt start_block, size_t typesize);
#define CopyContiguousBlockData(dest, source, npoints, nblocks, bs, start_block) \
  CopyContiguousBlockData_Generic(dest, source, npoints, nblocks, bs, start_block, sizeof(*(dest)))

PetscErrorCode CopyContiguousBlockData_Real(PetscReal *dest, PetscReal *source, PetscInt  npoints, PetscInt  nblocks, PetscInt  bs, PetscInt  start_block);
PetscErrorCode CopyArbitraryBlockData_Real(PetscReal dest[], PetscReal source[], PetscInt npoints, PetscInt nblocks, PetscInt bs, PetscInt blocks_to_copy[]);
PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscReal halton_sequence(PetscInt index, PetscInt base);
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode, PetscBool halton);
PetscErrorCode DMSwarmRemoveNPointsFromIndex(DM swarm, PetscInt nindices, PetscInt *indices);
PetscErrorCode DMSwarmPICSetCellId(DM sw);
PetscErrorCode DMSwarmMigrate_Push_DuplicatePoints(DM swarm, PetscInt point_start, PetscInt point_end);
#endif