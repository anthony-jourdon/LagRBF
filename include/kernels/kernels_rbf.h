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
  file:    kernels_rbf.h

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


#ifndef __LagRBF_kernels_rbf_h
#define __LagRBF_kernels_rbf_h

#include <petsc.h>

typedef enum {
  KERNEL_LINEAR=0,
  KERNEL_WENDLAND_C0,
  KERNEL_WENDLAND_C2,
  KERNEL_BUMP,
  KERNEL_MQ,
  KERNEL_IMQ,
  KERNEL_GAUSSIAN
} KernelType;

PetscReal KernelLinear(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC0(PetscReal r, PetscReal epsilon);
PetscReal KernelWendlandC2(PetscReal r, PetscReal epsilon);
PetscReal KernelBump(PetscReal r, PetscReal epsilon);
PetscReal KernelMultiquadric(PetscReal r, PetscReal epsilon);
PetscReal KernelInverseMultiquadric(PetscReal r, PetscReal epsilon);
PetscReal KernelGaussian(PetscReal r, PetscReal epsilon);

PetscErrorCode KernelGetFunction(KernelType ktype, PetscReal (**kernel_fn)(PetscReal,PetscReal));

#if defined(__AVX__)
#include "kernels/kernels_rbf_avx.h"
PetscErrorCode KernelGetFunction_AVX256(KernelType ktype, __m256d (**kernel_avx)(__m256d,__m256d));
#endif
#if defined(__AVX512F__)
#include "kernels/kernels_rbf_avx512.h"
PetscErrorCode KernelGetFunction_AVX512(KernelType ktype, __m512d (**kernel_avx)(__m512d,__m512d));
#endif

#endif