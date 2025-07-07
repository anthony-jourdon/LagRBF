#include <petsc.h>
#include "lagrangian_rbf.h"
#include "swarm_utils.h"
#include "kernels_rbf.h"
#include "swarm_interp.h"

PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  DM               swarm;
  PetscInt         int_order;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetIntegrationOrder(adv_ctx,&int_order));

  PetscCall(SwarmCreatePIC(pu,&swarm));
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pidx",1,PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",pu->npart,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL)); // fields to interpolate
  if (adv_ctx->scheme != RK1) {
    PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"v_adv",pu->dim*int_order,PETSC_REAL)); // velocity field interpolated
    PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"coor_init",pu->dim,PETSC_REAL)); // coordinates before advection
  }
  // May need to add another set of fields that need to be advected but not interpolated
  // PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pfields",ctx->n_pfields,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetCoordinatesFromOptions(PartitionOfUnity pu, DM sw, InsertMode mode)
{
  DM         da=NULL;
  PetscInt   m[]={0,0,1};
  PetscInt   d;
  PetscReal  O[]={0.0, 0.0, 0.0};
  PetscReal  L[]={1.0, 1.0, 1.0};
  PetscBool  found,spherical=PETSC_FALSE;
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetOptionsPrefix(sw,&prefix));
  d = pu->dim;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-npi",m,&d,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snpi not found. User must provide the number of interpolation points in each dimension.",prefix);
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%snpi must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-O",O,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sO must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-L",L,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sL must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetBool(NULL,prefix,"-spherical",&spherical,&found));
  if (spherical) {
    PetscCall(DMSwarmSetPointsSphericalCoordinates(sw,O,L,m,mode));
  } else {
    PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,mode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpLoadCoordinates(PartitionOfUnity pu, DM sw)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* For mpi implementation, test:
  - Get the total number of points in the swarm (the global size over all ranks)
  - Allocate the weights array with the total size
  - Get a map from local index to global index for the points in the swarm
  - Loop over the local points and compute the local weights and the local sum
  - Communicate the local sums to all ranks with MPI_Allreduce
  - Normalize the weights by dividing each local weight by the global sum of weights for that point
*/

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
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

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

PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx)
{
  PetscInt    p,npoints,max_local,max,np_local,np_global;
  PetscInt    *pidx;
  PetscMPIInt rank;
  MPI_Comm    comm;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PetscObjectGetComm((PetscObject)sw,&comm));
  PetscCall(MPI_Comm_rank(comm,&rank));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetField(sw, "pidx", NULL, NULL, (void **)&pidx));
  /* find max pid presently in the system */
  max_local = 0;
  for (p=0; p<npoints; p++) {
    PetscInt mp_idx = pidx[p]; 
    if ( mp_idx > max_local ) { max_local = mp_idx; }
  }
  PetscCall(MPI_Allreduce(&max_local,&max,1,MPI_LONG,MPI_MAX,comm));
  /*PetscPrintf(comm,"[[ %s ]] : max_pid = %" PetscInt_FMT "\n",PETSC_FUNCTION_NAME,max);*/
  max += 1;

  /* give particles a unique identifier */
  np_local = end_idx - start_idx;
  PetscCall(MPI_Scan(&np_local,&np_global,1,MPI_LONG,MPI_SUM,comm));
  //printf("rank %d : np_local = %" PetscInt_FMT ", np_global = %" PetscInt_FMT "\n",(int)rank,np_local,np_global);
  for (p=start_idx; p<end_idx; p++) {
    pidx[p] = max + (np_global-np_local) + (long int)(p-start_idx);
    //printf("assigning %" PetscInt_FMT " -> pid = %" PetscInt_FMT " \n", p, pidx[p] );
  }
  PetscCall(DMSwarmRestoreField(sw,"pidx",NULL,NULL,(void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw)
{
  DM       swarm;
  PetscInt npoints;
  PetscFunctionBeginUser;

  PetscCall(SwarmInterpCreate(ctx,&swarm));
  if (ctx->restart) {
    //PetscCall(SwarmInterpLoadCoordinates(pu,swarm));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Restart not implemented yet for Swarm Interpolation");
  } else {
    PetscCall(SwarmInterpSetCoordinatesFromOptions(ctx->pu,swarm,INSERT_VALUES));
  }
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm,0,npoints));
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpAddMarkersFromOptions(LagRBFCtx ctx, DM sw, InsertMode mode)
{
  PartitionOfUnity pu;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,sw,mode));
  PetscCall(SwarmInterpSetWeights(ctx,sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}