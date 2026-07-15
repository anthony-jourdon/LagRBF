#ifndef __LagRBF_swarm_output_h
#define __LagRBF_swarm_output_h

#include <petsc.h>
#include "partition_of_unity.h"

PetscErrorCode SwarmOutputFieldPetscIS_FromName(DM sw, const char *field_name, const char *directory, const char *prefix);
PetscErrorCode SwarmOutputFieldPetscVec_FromName(DM sw, const char *field_name, const char *directory, const char *prefix, PetscBool permute);
PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw, PetscInt nfields, const char *fields[], PetscBool permute);

#endif