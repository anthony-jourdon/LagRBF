#ifndef __softname_interpolation_avx_h
#define __softname_interpolation_avx_h

#include <petsc.h>
#include "cell_info.h"
#include "partition_of_unity.h"
#include "partition.h"
#include <immintrin.h>

PetscErrorCode PointwiseInterpolation_AVX256(
  PartitionOfUnity pu, Partition *partitions, CellInfo *cellinfo, 
  PetscInt nfields, PetscInt npoints_i, PetscInt *cellid_i,PetscReal *coords_i, 
  PetscReal *coords_d, PetscReal *rbf_d, PetscReal *weights, PetscReal *values_i
);

#endif