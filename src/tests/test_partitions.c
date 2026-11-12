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
  file:    test_partitions.c

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
//#include "partition_of_unity.h"
//#include "partition.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"

static double franke_function(double x, double y)
{
  /* Franke's function */
  /* f1 = 0.75 * np.exp(-((9 * x - 2)**2 + (9 * y - 2)**2) / 4)
     f2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49 - (9 * y + 1)**2 / 10)
     f3 = 0.5 * np.exp(-((9 * x - 7)**2 + (9 * y - 3)**2) / 4)
     f4 = 0.2 * np.exp(-((9 * x - 4)**2 + (9 * y - 7)**2))
     f = f1 + f2 + f3 - f4 
   */
  double value;
  
  value  =  0.75 * PetscExpReal(-((9.0 * x - 2.0) * (9.0 * x - 2.0) + (9.0 * y - 2.0) * (9.0 * y - 2.0)) / 4.0);
  value +=  0.75 * PetscExpReal(-((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0 - (9.0 * y + 1.0) * (9.0 * y + 1.0) / 10.0);
  value +=  0.5 * PetscExpReal(-((9.0 * x - 7.0) * (9.0 * x - 7.0) + (9.0 * y - 3.0) * (9.0 * y - 3.0)) / 4.0);
  value += -0.2 * PetscExpReal(-((9.0 * x - 4.0) * (9.0 * x - 4.0) + (9.0 * y - 7.0) * (9.0 * y - 7.0)));

  return value;
}

static PetscErrorCode PartitionTestGetters(PartitionOfUnity pu, Partition *partition)
{
  PetscInt i,npart;
  PetscFunctionBeginUser;

  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscPrintf(PETSC_COMM_WORLD,"Number of partitions: %" PetscInt_FMT "\n",npart);
  
  for (i=0; i<npart; i++) {
    Partition p = partition[i];
    Vec       X;
    PetscInt  k,ncells,ndata;
    PetscInt  *map,*data_idx;
    PetscReal epsilon;
    
    PartitionGetMap(p,&ncells,&map);
    PartitionGetDataPointsIdx(p,&ndata,&data_idx);
    PartitionGetCoefficientsVec(p,&X);
    PartitionGetEpsilon(p,&epsilon);
    PetscPrintf(PETSC_COMM_WORLD,"Partition[%" PetscInt_FMT "]: ncells=%" PetscInt_FMT ", ndata=%" PetscInt_FMT ", epsilon=%g\n",i,ncells,ndata,epsilon);
    if (ncells > 0) {
      for (k=0; k<ncells; k++) {
        PetscPrintf(PETSC_COMM_WORLD,"  map[%" PetscInt_FMT "] = %" PetscInt_FMT "\n",k,map[k]);
      }
    }
    if (ndata > 0) {
      for (k=0; k<ndata; k++) {
        PetscPrintf(PETSC_COMM_WORLD,"  data_idx[%" PetscInt_FMT "] = %" PetscInt_FMT "\n",k,data_idx[k]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataPoints(PartitionOfUnity pu, DM *sw)
{
  DM        da,swarm;
  PetscInt  dim,d;
  PetscInt  *m;
  PetscBool found;
  char      dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN],
            option_name[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetDimension(da, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)da), &swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmSetCellDM(swarm, da));

  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  /* create the field that will hold the data function values */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"data", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  /* Set coords */
  DMGetDimension(swarm,&dim);
  PetscMalloc1(dim,&m);
  d = dim;
  PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-n");
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,option_name,m,&d,&found));
  PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",dm_prefix);
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_USER,"%s option not found. Use it to specify the number of points in each direction.",option_name);
  PetscCheck(d == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,d);
/* Create particles and set coordinates (all in one) */
  PetscCall(DMGetBoundingBox(da,pu->O,pu->L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,pu->O,pu->L,m,INSERT_VALUES));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmSetDataFunction(DM sw)
{
  DMSwarmCellDM celldm;
  PetscReal     *coords,*data;
  PetscInt      Nfc,npoints,np;
  const char    **coordFields, *cellid;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  /* Get points coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,"data",NULL,NULL,(void **)&data));

  for (np=0; np<npoints; np++) {
    /* Get the coordinates of the particle */
    PetscReal x = coords[2*np];
    PetscReal y = coords[2*np+1];
    /* Compute the value of the function at that point */
    data[np] = franke_function(x,y);
  }
  /* Set the field with the computed values */
  PetscCall(DMSwarmRestoreField(sw,"data",NULL,NULL,(void **)&data));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmCreateInterpolationPoints(PartitionOfUnity pu, DM *sw)
{
  DM        da,swarm;
  PetscInt  m[] = {16, 16}, ndim, d, si,sj,nx,ny,dim;
  PetscReal O[3],L[3];
  PetscBool found;
  char     dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetDimension(da, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)da), &swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmSetCellDM(swarm, da));
  /* Get number of partitions */
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  /* create the field that will hold the data function values */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"franke", 1,     PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"error",  1,     PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"w_j",    nx*ny, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  /* Set coords */
  PetscCall(DMGetDimension(swarm,&ndim));
  PetscCall(DMGetDimension(swarm,&d));
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-nppc",m,&d,&found));
  if (found) PetscCheck(d == ndim, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-%snppc must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "", dm_prefix, ndim, d);
  /* Create particles and set coordinates (all in one) */
  PetscCall(DMGetBoundingBox(da,O,L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetWeights(PartitionOfUnity pu, Partition *partitions, DM swarm_data, DM swarm_interp)
{
  DMSwarmCellDM celldm;
  DM            da,cda;
  Vec           coord;
  PetscReal     *LA_coords,*icoords,*weights,*sum_w;
  PetscInt      i,si,sj,nx,ny,nip,dim;
  PetscInt      np_interp,npu;
  const char    **icoordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  /* PU info */
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  dim = pu->dim;
  /* Interpolation points info */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&np_interp)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(swarm_interp, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &icoordFields));
  PetscCall(DMSwarmGetField(swarm_interp, icoordFields[0], NULL, NULL, (void **)&icoords));
  PetscCall(DMSwarmGetField(swarm_interp, "w_j", &npu, NULL, (void **)&weights));
  PetscCheck(npu == nx*ny, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of weigths must equal the number of partitions of unity, found: %" PetscInt_FMT "", npu);

  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords)); // I think this is rank local

  /* Sum for weights */
  PetscCall(PetscCalloc1(np_interp,&sum_w)); // local rank => Need to talk with Dave about parallelism
  /* Loop over points */
  for (nip=0; nip<np_interp; nip++) {
    /* Initialize sum of weights */
    /* Loop over partitions of unity */
    for (i=0; i<pu->npart; i++) {
      PetscInt  d,puidx;
      PetscReal *pu_centre;
      PetscReal dist = 0.0;

      puidx = partitions[i]->idx; // partition index, rank local (not the global index)
      pu_centre = &LA_coords[dim*puidx];
      for (d=0; d<dim; d++) { 
        dist += (icoords[dim*nip+d] - pu_centre[d]) * (icoords[dim*nip+d] - pu_centre[d]); 
      }
      dist = PetscSqrtReal(dist);
      /* Compute the weight */
      weights[nip*nx*ny + puidx] = KernelWendlandC2(dist,partitions[i]->epsilon);
      sum_w[nip] += weights[nip*nx*ny + puidx];
    }
  }

  // Here should perform communication to get the total sum of weights and send it to all processes 
  // but I am not sure if all processes have the same number of points so for now no parallelism
  /* Normalize weights */
  for (nip=0; nip<np_interp; nip++) {
    for (i=0; i<pu->npart; i++) {
      PetscInt puidx = partitions[i]->idx; // partition index, rank local (not the global index)
      if (sum_w[nip] > 0.0) {
        /* Normalize the weights */
        weights[nip*pu->npart + puidx] /= sum_w[nip];
      } else {
        /* If the sum of weights is zero, set the weight to zero */
        weights[nip*pu->npart + puidx] = 0.0;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swarm_interp,icoordFields[0],NULL,NULL,(void **)&icoords));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  /* Free memory */
  PetscCall(PetscFree(sum_w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLocalCoefficients(
  Partition p,
  PetscReal data_coords[],
  PetscReal fval[]
)
{
  Vec       RHS,X;
  PetscInt  ndata;
  PetscInt  *data_idx;
  PetscReal epsilon;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
  /* Set the size of the vector for the coefficients */
  PetscCall(PartitionGetCoefficientsVec(p,&X));
  PetscCall(VecSetSizes(X,PETSC_DECIDE,ndata));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecZeroEntries(X));
  PetscCall(PartitionGetEpsilon(p,&epsilon));
  /* Create, assemble and factor the RBF matrix */
  PetscCall(PartitionSetUpRBFMatrix(p,2,data_coords,KernelWendlandC2));
  /* Create and fill the RHS vector */
  PetscCall(PartitionSetVecRHS(p,fval,&RHS));
  PetscCall(MatSolve(p->A,RHS,X));
  /* Clean-up */
  PetscCall(VecDestroy(&RHS));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLocalCoefficients_KSP(
  Partition p,
  PetscReal data_coords[],
  PetscReal fval[]
)
{
  Mat       RBF_data;
  Vec       RHS,X;
  KSP       ksp;
  PetscInt  ndata;
  PetscInt  *data_idx;
  PetscReal epsilon;
  PetscFunctionBeginUser;

  PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"rbf_"));
  PetscCall(KSPSetTolerances(ksp,1e-10,PETSC_CURRENT,PETSC_CURRENT,PETSC_CURRENT));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
  /* Set the size of the vector for the coefficients */
  PetscCall(PartitionGetCoefficientsVec(p,&X));
  PetscCall(VecSetSizes(X,PETSC_DECIDE,ndata));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecZeroEntries(X));
  PetscCall(PartitionGetEpsilon(p,&epsilon));
  /* Create, assemble the RBF matrix */
  PetscCall(PartitionAssembleRBFMatrix(p,2,data_coords,KernelWendlandC2,&RBF_data));
  /* Create and fill the RHS vector */
  PetscCall(PartitionSetVecRHS(p,fval,&RHS));

  PetscCall(KSPSetOperators(ksp,RBF_data,RBF_data));
  PetscCall(KSPSolve(ksp,RHS,X));
  /* Clean-up */
  PetscCall(MatDestroy(&RBF_data));
  PetscCall(VecDestroy(&RHS));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveInterpolationCoefficients(
  PartitionOfUnity pu,
  Partition *partition,
  DM swarm_data
)
{
  DMSwarmCellDM celldm;
  DM            da;
  PetscInt      i,si,sj,nx,ny,npoints;
  PetscReal     *data_coords,*data;
  const char    **dcoordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));

  /* Data points */
  PetscCall(DMSwarmSortGetAccess(swarm_data));
  PetscCall(DMSwarmGetLocalSize(swarm_data,&npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &dcoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_data, dcoordFields[0], NULL, NULL, (void **)&data_coords));
  PetscCall(DMSwarmGetField(swarm_data,"data",NULL,NULL,(void **)&data));

  /* Loop over partitions */
  for (i=0; i<pu->npart; i++) {
    Partition p = partition[i];
    if (p->ndata == 0) continue; // no data points in this partition, skip
    PetscCall(SolveLocalCoefficients(p,data_coords,data));
  }
  /* Restore and cleanup */
  PetscCall(DMSwarmSortRestoreAccess(swarm_data));
  PetscCall(DMSwarmRestoreField(swarm_data,dcoordFields[0],NULL,NULL,(void **)&data_coords));
  PetscCall(DMSwarmRestoreField(swarm_data,"data",NULL,NULL,(void **)&data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointwiseInterpolation(
  DM da, 
  DM swarm_data,
  DM swarm_interp,
  Partition *partition
)
{
  DMSwarmCellDM icelldm,dcelldm;
  DM            cda;
  Vec           coord;
  PetscInt      *interp_cellid;
  PetscInt      p,k,si,sj,nx,ny,nel,nen,interp_npoints;
  PetscReal     *interp_coords,*data_coords,*weights,*LA_coords,*interp_values;
  const PetscInt *elidx;
  const char    **icoordFields,**dcoordFields,*icellidFields;
  PetscFunctionBeginUser;

  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&interp_npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_interp, &icelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(icelldm, NULL, &icoordFields));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(icelldm,&icellidFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_interp, icoordFields[0], NULL, NULL, (void **)&interp_coords));
  PetscCall(DMSwarmGetField(swarm_interp, icellidFields, NULL, NULL, (void **)&interp_cellid));
  PetscCall(DMSwarmGetField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swarm_interp,"franke",NULL,NULL,(void **)&interp_values));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data, &dcelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(dcelldm, NULL, &dcoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_data, dcoordFields[0], NULL, NULL, (void **)&data_coords));

  for (p=0; p<interp_npoints; p++) {
    PetscInt cidx = interp_cellid[p]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    interp_values[p] = 0.0;
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition part;
      Vec       X;
      PetscInt  nd,puidx,ndata;
      PetscInt  *data_idx;
      PetscReal interpolant;
      PetscReal *coeffs;

      puidx = elidx[nen*cidx + k];
      part  = partition[puidx];
      PetscCall(PartitionGetDataPointsIdx(part,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      PetscCall(PartitionGetCoefficientsVec(part,&X));
      /* Get the coefficients of that partition */
      PetscCall(VecGetArray(X,&coeffs));
      interpolant = 0.0;
      /* Replace the interpolation on scalar by linear algebra */
      for (nd=0; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;

        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<2; d++) {
          dist += (interp_coords[2*p + d] - data_coords[2*data_idx[nd] + d]) * (interp_coords[2*p + d] - data_coords[2*data_idx[nd] + d]);
        }
        dist = PetscSqrtReal(dist);
        rbf  = KernelWendlandC2(dist,part->epsilon);
        rbf *= coeffs[nd]; // multiply the RBF value by the coefficient
        interpolant += rbf; // accumulate the value for the interpolation point
      }
      /* Normalize the interpolant with the weights */
      interp_values[p] += interpolant * weights[p*nx*ny + puidx];
      /* Restore the coefficients array */
      PetscCall(VecRestoreArray(X,&coeffs));
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_interp,icoordFields[0],NULL,NULL,(void **)&interp_coords));
  PetscCall(DMSwarmRestoreField(swarm_interp,icellidFields,NULL,NULL,(void **)&interp_cellid));
  PetscCall(DMSwarmRestoreField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swarm_interp,"franke",NULL,NULL,(void **)&interp_values));
  PetscCall(DMSwarmRestoreField(swarm_data,dcoordFields[0],NULL,NULL,(void **)&data_coords));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_data,swarm_interp;
  PetscReal        overlap,radius,epsilon;
  PetscInt         dim,M,N,m,n,k;
  PetscBool        exists;
  const char       *output_dir = "output";
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }
  
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  PetscCall(DMDAGetInfo(pu->dap,&dim,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  overlap = 1.5;
  radius  = overlap / PetscSqrtReal(M*N);
  epsilon = 1.0 / radius;
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  PetscCall(CreateDataPoints(pu,&swarm_data));
  PetscCall(DMSwarmSetDataFunction(swarm_data));
  PetscCall(DMSwarmCreateInterpolationPoints(pu,&swarm_interp));
  PetscCall(SetWeights(pu,partitions,swarm_data,swarm_interp));
  PetscCall(PartitionFillDataPointsIdx_All(partitions,pu->npart,swarm_data));
  //PetscCall(PartitionTestGetters(pu,partitions));

  PetscCall(SolveInterpolationCoefficients(pu,partitions,swarm_data));
  PetscCall(PointwiseInterpolation(pu->dap,swarm_data,swarm_interp,partitions));

  /* Output */
  PetscCall(PartitionOfUnityMeshView(pu));
  /* output swarm data */
  {
    char fname[PETSC_MAX_PATH_LEN];
    const char *fields_names[]= {"data"};
    PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s/swarm_data_ptwise.xmf", output_dir);
    PetscCall(DMSwarmViewFieldsXDMF(swarm_data,fname,1,fields_names));
  }
  /* output interpolated data */
  {
    char fname[PETSC_MAX_PATH_LEN];
    const char *fields_names[]= {"franke"};
    PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s/swarm_interp_ptwise.xmf", output_dir);
    PetscCall(DMSwarmViewFieldsXDMF(swarm_interp,fname,1,fields_names));
  }

  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}



int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}