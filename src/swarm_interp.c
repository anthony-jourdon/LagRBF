#include <petsc.h>
#include "lagrangian_rbf.h"
#include "swarm_utils.h"

static PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               swarm;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));

  PetscCall(SwarmCreatePIC(pu,&swarm));
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",pu->npart,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  // May need to add another set of fields that need to be advected but not interpolated
  // PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pfields",ctx->n_pfields,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpSetUniformCoordinates(PartitionOfUnity pu, DM sw)
{
  DM         da;
  PetscInt   *m;
  PetscInt   d;
  PetscReal  O[3],L[3];
  PetscBool  found;
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetBoundingBox(da,O,L));
  PetscCall(DMGetOptionsPrefix(da,&prefix));
  d = pu->dim;
  PetscCall(PetscMalloc1(d,&m));
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-npi",m,&d,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snpi not found. User must provide the number of interpolation points in each dimension.",prefix);
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%snpi must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,INSERT_VALUES));

  PetscFree(m);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpLoadCoordinates(PartitionOfUnity pu, DM sw)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpSetWeights_SEQ(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DMSwarmCellDM    celldm;
  DM               da,cda;
  Vec              coord;
  PetscInt         i,np,dim,npoints,npart,nn;
  PetscReal        *pcoords,*weights,*LA_coords,*sum_w;
  const char      **coordFields;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(PUGetNPartitions(pu,&npart));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCheck(nn == npart, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of weigths must equal the number of partitions of unity: %" PetscInt_FMT ", found: %" PetscInt_FMT "",npart,nn);

  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords));

  /* Sum for weights */
  PetscCall(PetscCalloc1(npoints,&sum_w));
  /* Loop over points */
  for (np=0; np<npoints; np++) {
    /* Initialize sum of weights */
    /* Loop over partitions of unity */
    for (i=0; i<npart; i++) {
      Partition p = partitions[i]; // current partition
      PetscReal epsilon,dist;
      PetscInt  d,puidx;
      PetscReal *pu_centre;
      
      PetscCall(PartitionGetIndex(p,&puidx));
      PetscCall(PartitionGetEpsilon(p,&epsilon));
      pu_centre = &LA_coords[dim*puidx];
      dist = 0.0;
      for (d=0; d<dim; d++) { 
        dist += (pcoords[dim*np + d] - pu_centre[d]) * (pcoords[dim*np + d] - pu_centre[d]); 
      }
      dist = PetscSqrtReal(dist);
      /* Compute the weight */
      weights[np*npart + puidx] = KernelWendlandC2(dist,epsilon);
      sum_w[np] += weights[np*npart + puidx];
    }
  }

  // Here should perform communication to get the total sum of weights and send it to all processes 
  // but I am not sure if all processes have the same number of points so for now no parallelism
  /* Normalize weights */
  for (np=0; np<npoints; np++) {
    for (i=0; i<npart; i++) {
      Partition p = partitions[i]; // current partition
      PetscInt  puidx;

      PetscCall(PartitionGetIndex(p,&puidx));
      if (sum_w[np] > 0.0) {
        /* Normalize the weights */
        weights[np*npart + puidx] /= sum_w[np];
      } else {
        /* If the sum of weights is zero, set the weight to zero */
        weights[np*npart + puidx] = 0.0;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  PetscCall(PetscFree(sum_w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw)
{
  PetscFunctionBeginUser;
  /* For now we only support sequential */
  PetscCall(SwarmInterpSetWeights_SEQ(ctx,sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               swarm;
  PetscFunctionBeginUser;

  PetscCall(SwarmInterpCreate(ctx,&swarm));
  if (ctx->restart) {
    //PetscCall(SwarmInterpLoadCoordinates(pu,swarm));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Restart not implemented yet for Swarm Interpolation");
  } else {
    PetscCall(SwarmInterpSetUniformCoordinates(ctx->pu,swarm));
  }
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}