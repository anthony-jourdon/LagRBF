#ifndef __softname_interface_utils_h
#define __softname_interface_utils_h

#include "lagrangian_rbf.h"

PetscErrorCode FillSwarm(LagRBFCtx ctx, DM sw, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx);
PetscErrorCode AssembleArray_Allreduce(PetscReal *local_array, PetscInt nlocal, PetscReal  *global_array, PetscInt nglobal, PetscInt64 *gidx, PetscInt nfields, MPI_Comm comm);

#endif