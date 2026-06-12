#ifndef __softname_interface_interp_h
#define __softname_interface_interp_h

#include "lagrangian_rbf.h"

LAGRBF_EXPORT void lagrbf_interpolate(
  PetscInt   ndata_local,
  PetscReal *data_coords,
  PetscReal *data_values,
  PetscInt   npoints_local,
  PetscReal *points_coords,
  PetscReal *points_values,
  PetscInt   npoints_global,
  char      *options_file
);

#endif