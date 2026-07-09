#ifndef __softname_interface_utils_h
#define __softname_interface_utils_h

#include "lagrangian_rbf.h"

PetscErrorCode FillSwarm(LagRBFCtx ctx, DM sw, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx);
PetscErrorCode AssembleArray_Allreduce(const void *local_array, PetscInt nlocal, void *global_array, PetscInt nglobal, const PetscInt64 *gidx, PetscInt nfields, PetscDataType array_type, MPI_Comm comm);
PetscErrorCode Interface_Output_Allreduce(DM sw, PetscInt np_global, PetscReal *global_coor, PetscReal *global_fields, PetscInt64 *global_pidx, const char *fieldname);
PetscErrorCode InterfacePetscErrorPrinter(PetscErrorCode ierr, const char *function_name);
PetscErrorCode FillInterpSwarm(LagRBFCtx ctx, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx);
PetscErrorCode FillDataSwarm(LagRBFCtx ctx, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx);

LAGRBF_EXPORT PetscErrorCode lagrbf_fill_swarm(
  void       *ctx,
  PetscBool   is_data,
  PetscInt    npoints,
  PetscReal  *coords,
  PetscReal  *values,
  PetscInt64 *pidx
);

LAGRBF_EXPORT PetscErrorCode lagrbf_output(
  void       *ctx,
  PetscBool  is_data,
  PetscInt   npoints,
  PetscReal  *coords,
  PetscReal  *fields,
  PetscInt64 *pidx,
  const char *fieldname
);

#endif