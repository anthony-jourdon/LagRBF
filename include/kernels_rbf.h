
#ifndef __softname_kernels_rbf_h
#define __softname_kernels_rbf_h

#include <petsc.h>

PetscReal KernelLinear(PetscReal r, PetscReal epsilon);
PetscReal KernelGaussian(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC0(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC2(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC4(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC6(PetscReal r, PetscReal epsilon);

#endif