#ifndef __softname_swarm_load_h
#define __softname_swarm_load_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmLoadVecFromFile(DM sw, const char *field_name, const char *filename, MPI_Comm comm, PetscBool resize);
PetscErrorCode SwarmLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmLoadFromFiles(LagRBFCtx ctx, DM sw, const char *dirname, const char **fields_name, PetscInt nfields);

#endif