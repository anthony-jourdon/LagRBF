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
  file:    interpolation.h

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

#ifndef __LagRBF_interpolation_h
#define __LagRBF_interpolation_h

#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode RBFInterpolation(LagRBFCtx ctx, DM swi, DM swd, const char *fieldname);

#if defined(__AVX512F__)
PetscErrorCode PointwiseInterpolation_AVX512(
  PartitionOfUnity pu,
  Partition        *partitions,
  CellInfo         *cellinfo,
  KernelType       ktype,
  PetscInt         nfields,
  PetscInt         npoints_i,
  PetscInt         *cellid_i,
  PetscReal        *coords_i,
  PetscReal        *coords_d,
  PetscReal        *rbf_d,
  PetscReal        *weights,
  PetscReal        *values_i
);
#endif

#if defined(__AVX__)
PetscErrorCode PointwiseInterpolation_AVX256(
  PartitionOfUnity pu,
  Partition *partitions,
  CellInfo *cellinfo,
  KernelType ktype,
  PetscInt nfields,
  PetscInt npoints_i,
  PetscInt *cellid_i,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *rbf_d,
  PetscReal *weights,
  PetscReal *values_i
);
#endif

#endif