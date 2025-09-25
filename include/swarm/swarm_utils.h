#ifndef __softname_swarm_utils_h
#define __softname_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmCopyBlockData_Real(DM swarm, PetscInt nblocks, PetscInt blocks_to_copy[], const char *field_name, PetscReal **copied_blocks);
PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode);

#endif