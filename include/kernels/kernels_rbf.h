
#ifndef __softname_kernels_rbf_h
#define __softname_kernels_rbf_h

#include <petsc.h>
#if defined(__AVX__)
#include "kernels/kernels_rbf_avx.h"
#endif

typedef enum {
  KERNEL_LINEAR=0,
  KERNEL_WENDLAND_C0,
  KERNEL_WENDLAND_C2,
  KERNEL_WENDLAND_C4,
  KERNEL_WENDLAND_C6
} KernelType;

PetscReal KernelLinear(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC0(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC2(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC4(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC6(PetscReal r, PetscReal epsilon);

#endif