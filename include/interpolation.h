#ifndef __softname_interpolation_h
#define __softname_interpolation_h

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