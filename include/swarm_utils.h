#ifndef __softname_swarm_utils_h
#define __softname_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscErrorCode SwarmOutputCoordsPetscVec(DM sw, const char *directory, const char *prefix);
PetscErrorCode SwarmOutputFieldsPetscVec(DM sw, const char *directory, const char *prefix);
PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw);
#endif