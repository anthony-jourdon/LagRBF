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
  file:    partition_solve.h

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

#ifndef __LagRBF_partition_solve_h
#define __LagRBF_partition_solve_h

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition.h"
#include "kernels/kernels_rbf.h"

PetscErrorCode PartitionCreateRBFMatrix(Partition p);
PetscErrorCode PartitionAssembleRBFMatrix(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype);
#if defined(__AVX__)
PetscErrorCode PartitionAssembleRBFMatrix_AVX256(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype);
#endif
#if defined(__AVX512F__)
PetscErrorCode PartitionAssembleRBFMatrix_AVX512(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype);
#endif
PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B);
PetscErrorCode PartitionSolveInterpolationCoefficients_SinglePartition(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal *values);

/* Cholesky Factor methods */
PetscErrorCode PartitionSetUpRBFMatrix_Factor(LagRBFCtx ctx, Partition p, PetscReal *coords);
PetscErrorCode PartitionSolveLocalCoefficientsFactor(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[]);

/* KSP re-used methods */
PetscErrorCode PartitionSetUpRBFMatrix_KSP(LagRBFCtx ctx, Partition p, PetscReal *coords);
PetscErrorCode PartitionSolveLocalCoefficientsKSP(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[]);
PetscErrorCode PartitionKSPDestroyCtx(Partition p);

/* KSP re-assembled will go here */
PetscErrorCode PartitionSolveLocalCoefficientsKSP_ASMB(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[]);

#endif