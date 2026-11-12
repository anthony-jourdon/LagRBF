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
  file:    partition_solve.c

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

#include <petsc.h>
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "kernels/kernels_rbf.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_Scalar;
PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;

PetscErrorCode PartitionCreateRBFMatrix(Partition p)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  PetscCheck(!p->A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix already created.");
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&p->A));
  PetscCall(MatSetFromOptions(p->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionAssembleRBFMatrix_Scalar(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype)
{
  Mat       A;
  PetscInt  i,j,d,npoints;
  PetscInt  *pidx;
  PetscReal *rbf;
  PetscReal (*kernel_fn)(PetscReal,PetscReal) = NULL;
  PetscFunctionBeginUser;

  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix_Scalar,0,0,0,0));
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(MatDenseGetArray(A,&rbf));
  /* Fill the matrix */
  for (i=0; i<npoints; i++) {
    for (j=i; j<npoints; j++) {
      PetscReal dist = 0.0;
      /* Compute the distance between points */
      for (d=0; d<dim; d++) {
        dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
      }
      dist = PetscSqrtReal(dist);
      /* Fill the upper triangular part with the RBF kernel */
      rbf[npoints*i + j] = kernel_fn(dist,epsilon);
      /* Copy the value to the lower triangular part (diagonal already filled) */
      if (i != j) {
        rbf[npoints*j + i] = rbf[npoints*i + j];
      }
    }
  }
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix_Scalar,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionAssembleRBFMatrix(Partition p, PetscInt dim, PetscReal epsilon, PetscReal *coords, KernelType ktype)
{
  PetscFunctionBeginUser;
#if defined(__AVX512F__)
  PetscCall(PartitionAssembleRBFMatrix_AVX512(p,dim,epsilon,coords,ktype));
#elif defined(__AVX__)
  PetscCall(PartitionAssembleRBFMatrix_AVX256(p,dim,epsilon,coords,ktype));
#else
  PetscCall(PartitionAssembleRBFMatrix_Scalar(p,dim,epsilon,coords,ktype));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSolveInterpolationCoefficients_SinglePartition(LagRBFCtx ctx, Partition p, PetscInt nfields, PetscReal *values)
{
  PartitionOfUnity pu;
  PetscInt npoints;
  PetscFunctionBeginUser;
  /* Partition is not setup, which means it is not needed, skip */
  if (!p->setup) { PetscFunctionReturn(PETSC_SUCCESS); }
  /* Partition as already been solved for, we don't need to re-solve */
  if (p->solved)  { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
  /* Partition does not contain any data points, skip */
  if (npoints == 0) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  /* Solve for the interpolation coefficients in this partition */
  PetscCall(pu->psolve(ctx,p,nfields,values));
  p->solved = PETSC_TRUE; // mark as solved
  PetscFunctionReturn(PETSC_SUCCESS);
}