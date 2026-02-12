#ifndef __softname_partition_solve_h
#define __softname_partition_solve_h

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