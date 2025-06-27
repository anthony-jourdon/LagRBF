#ifndef __softname_swarm_utils_h
#define __softname_swarm_utils_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw);
PetscErrorCode SwarmOutputFieldPetscVec_FromName(DM sw, const char *field_name, const char *directory, const char *prefix);
PetscErrorCode SwarmOutputCoordsPetscVec(DM sw, const char *directory, const char *prefix);
PetscErrorCode SwarmOutputFieldsPetscVec(DM sw, const char *directory, const char *prefix);
PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw, PetscInt nfields, const char *fields[]);
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode);

#endif