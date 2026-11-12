/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    interface_utils.h

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#ifndef __LagRBF_interface_utils_h
#define __LagRBF_interface_utils_h

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

LAGRBF_EXPORT PetscErrorCode lagrbf_reset_coefficients(void *ctx);

#endif