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
  file:    interpolation.c

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
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "cell_info.h"
#include "kernels/kernels_rbf.h"
#include "interpolation.h"

PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;

PetscErrorCode PointwiseInterpolation_Scalar(
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
)
{
  DM             da;
  PetscInt       k,np,dim,nel,nen;
  PetscInt       *gtol_elidx;
  const PetscInt *elidx;
  PetscReal      epsilon;
  PetscReal      *interpolant;
  PetscReal      (*kernel_fn)(PetscReal,PetscReal);
  PetscLogDouble time[2];
  long int       npoints_local,npoints_global;
  double         time_local,time_max;
  PetscFunctionBeginUser;
  /* Logging */
  PetscCall(PetscTime(&time[0]));
  PetscCall(PetscLogEventBegin(LAGRBF_PointwiseInterpolation_Scalar,0,0,0,0));
  /* Select the appropriate kernel */
  PetscCall(KernelGetFunction(ktype,&kernel_fn));
  /* Get the parallel-sequtial DMDA elements mapping */
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  PetscCall(PUGetShapeFactor(pu,&epsilon));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  for (np=0; np<npoints_i; np++) {
    CellInfo  ci;
    PetscInt  f,nd,cidx,ci_ndata,local_cidx;
    PetscInt  *ci_data_idx;
    
    /* Cell index of the particle */
    cidx       = cellid_i[np];     // parallel DMDA
    local_cidx = gtol_elidx[cidx]; // local sequential DMDA
    /* Cell info of this cell */
    ci = cellinfo[local_cidx];
    PetscCall(CellInfoGetDataPointsIdx(ci,&ci_ndata,&ci_data_idx));
    if (ci_ndata == 0) continue; // no data points in this cell neighbourhood, skip
    /* Precompute the RBF values for all data points connected to that cell */
    for (nd=0; nd<ci_ndata; nd++) {
      PetscInt  d,dpidx;
      PetscReal dist;
      dpidx = ci_data_idx[nd];
      /* Compute the RBF value for the interpolation point */
      dist = 0.0;
      for (d=0; d<dim; d++) {
        PetscReal dx = coords_i[dim*np + d] - coords_d[dim*dpidx + d];
        dist += dx * dx;
      }
      rbf_d[dpidx] = kernel_fn(PetscSqrtReal(dist), epsilon);
    }
    /* Initialize the interpolation value for the point p */
    PetscCall(PetscArrayzero(&values_i[nfields*np],nfields));
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  puidx,ndata;
      PetscInt  *data_idx;
      PetscReal *coeffs;
      PetscReal weight;

      /* Get the partition index */
      puidx = elidx[nen*local_cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      PetscCall(PetscArrayzero(interpolant,nfields));
      for (nd=0; nd<ndata; nd++) {
        PetscReal rbf = rbf_d[ data_idx[nd] ];
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      /* Get the weight */
      weight = weights[nen*np + k];
      for (f=0; f<nfields; f++) {
        /* Normalize the interpolant with the weights */
        values_i[nfields*np + f] += interpolant[f] * weight;
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscLogEventEnd(LAGRBF_PointwiseInterpolation_Scalar,0,0,0,0));
  PetscCall(PetscTime(&time[1]));
  /* Need to perform a reduction to gather the total number of points */
  npoints_local = (long int)npoints_i;
  PetscCallMPI(MPI_Allreduce(&npoints_local,&npoints_global,1,MPI_LONG,MPI_SUM,PETSC_COMM_WORLD));
  time_local = (double)time[1]-(double)time[0];
  PetscCallMPI(MPI_Allreduce(&time_local,&time_max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  /* Perform reduction on time to get the max */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %ld points in %1.4e seconds\n",PETSC_FUNCTION_NAME,npoints_global,time_max));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  RBFInterpolation - Interpolate values of one field from data swarm points to interpolation swarm points

  Collective

  Input Parameters:
+ ctx (LagRBFCtx)         - the LagRBF context
. swi (DM)                - interpolation swarm (target points)
. swd (DM)                - data swarm (source points)
- fieldname (const char*) - name of the field to interpolate

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  Backend is selected at compile time:
  ``__AVX512F__``: AVX-512 implementation
  ``__AVX__``: AVX-256 implementation
  ``fallback``: Scalar implementation

.seealso: DM, DMSWARM, DMSwarmGetField(), DMSwarmRestoreField(), PetscErrorCode
@*/
PetscErrorCode RBFInterpolation(LagRBFCtx ctx, DM swi, DM swd, const char *fieldname)
{
  DMSwarmCellDM    celldm_i,celldm_d;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *cellinfo = NULL;
  KernelType       ktype;
  PetscInt         npoints_i,nfields;
  PetscInt         *cellid_i;
  PetscReal        *coords_i,*coords_d,*weights,*values_i,*rbf_d;
  const char       **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscFunctionBeginUser;
  
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&cellinfo));
  PetscCall(LagRBFGetKernelType(ctx,&ktype));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_i,NULL,&coordFieldName_i));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coordFieldName_i[0], NULL, NULL, (void **)&coords_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMSwarmGetField(swi, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swi, fieldname,&nfields,NULL,(void **)&values_i));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd,&celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d,NULL,&coordFieldName_d));
  /* Get coordinates */
  PetscCall(DMSwarmGetField(swd, coordFieldName_d[0], NULL, NULL, (void **)&coords_d));
  PetscCall(DMSwarmGetField(swd, "rbf", NULL, NULL, (void **)&rbf_d));

  /* Perform the interpolation */
#if defined(__AVX512F__)
  PetscCall(PointwiseInterpolation_AVX512(
    pu,
    partitions,
    cellinfo,
    ktype,
    nfields,
    npoints_i,
    cellid_i,
    coords_i,
    coords_d,
    rbf_d,
    weights,
    values_i
  ));
#elif defined(__AVX__)
  PetscCall(PointwiseInterpolation_AVX256(
    pu,
    partitions,
    cellinfo,
    ktype,
    nfields,
    npoints_i,
    cellid_i,
    coords_i,
    coords_d,
    rbf_d,
    weights,
    values_i
  ));
#else
  PetscCall(PointwiseInterpolation_Scalar(
    pu,
    partitions,
    cellinfo,
    ktype,
    nfields,
    npoints_i,
    cellid_i,
    coords_i,
    coords_d,
    rbf_d,
    weights,
    values_i
  ));
#endif
  
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,fieldname,&nfields,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMSwarmRestoreField(swd,"rbf",NULL,NULL,(void **)&rbf_d));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}