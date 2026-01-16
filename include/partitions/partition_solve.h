#ifndef __softname_partition_solve_h
#define __softname_partition_solve_h

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition.h"

PetscErrorCode PartitionCreateRBFMatrix(Partition p);
PetscErrorCode PartitionAssembleRBFMatrix(Partition p,PetscInt dim,PetscReal epsilon,PetscReal coords[],PetscReal (*kernel_fn)(PetscReal,PetscReal));
PetscErrorCode PartitionMatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B);
PetscErrorCode PartitionSolveInterpolationCoefficients(LagRBFCtx ctx, DM swarm, const char *fieldname);

/* Cholesky Factor methods */
PetscErrorCode PartitionSetUpRBFMatrix_Factor(LagRBFCtx ctx, Partition p, PetscReal *coords);
PetscErrorCode PartitionSolveLocalCoefficientsFactor(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[]);

/* KSP re-used methods */
PetscErrorCode PartitionSetUpRBFMatrix_KSP(LagRBFCtx ctx, Partition p, PetscReal *coords);
PetscErrorCode PartitionSolveLocalCoefficientsKSP(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal fval[]);
PetscErrorCode PartitionKSPDestroyCtx(Partition p);

/* KSP re-assembled will go here */

#endif