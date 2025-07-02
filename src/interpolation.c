#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "partition.h"
#include "partition_solve.h"
#include "kernels_rbf.h"
#if defined(__AVX__)
#include "interpolation_avx.h"
#endif

PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;

PetscErrorCode PointwiseInterpolation_Scalar(
  PartitionOfUnity pu,
  Partition *partitions,
  PetscInt nfields,
  PetscInt npoints_i,
  PetscInt *cellid_i,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *weights,
  PetscReal *values_i
)
{
  DM             da;
  PetscInt       k,np,dim,npart,nel,nen;
  const PetscInt *elidx;
  PetscReal      *interpolant;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;
  /* Logging */
  PetscCall(PetscTime(&time[0]));
  PetscCall(PetscLogEventBegin(LAGRBF_PointwiseInterpolation_Scalar,0,0,0,0));

  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Allocate memory for the interpolant */
  PetscCall(PetscCalloc1(nfields,&interpolant));
  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) { values_i[f*npoints_i + np] = 0.0; }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata;
      PetscInt  *data_idx;
      PetscReal *coeffs;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      for (f=0; f<nfields; f++) { interpolant[f] = 0.0;}
      for (nd=0; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          PetscReal dx = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += dx * dx;
        }
        rbf = KernelWendlandC2(PetscSqrtReal(dist),p->epsilon);
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      for (f=0; f<nfields; f++) {
        /* Normalize the interpolant with the weights */
        values_i[f*npoints_i + np] += interpolant[f] * weights[np*npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscLogEventEnd(LAGRBF_PointwiseInterpolation_Scalar,0,0,0,0));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RBFInterpolation(LagRBFCtx ctx, DM swi, DM swd, const char *fieldname)
{
  DMSwarmCellDM    celldm_i,celldm_d;
  PartitionOfUnity pu;
  Partition        *partitions;
  PetscInt         npoints_i,nfields;
  PetscInt         *cellid_i;
  PetscReal        *coords_i,*coords_d,*weights,*values_i;
  const char       **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscFunctionBeginUser;
  
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
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

  /* Perform the interpolation */
#if defined(__AVX__)
  PetscCall(PointwiseInterpolation_AVX256(pu,partitions,nfields,npoints_i,cellid_i,coords_i,coords_d,weights,values_i));
#else
  PetscCall(PointwiseInterpolation_Scalar(pu,partitions,nfields,npoints_i,cellid_i,coords_i,coords_d,weights,values_i));
#endif
  
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,fieldname,&nfields,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}