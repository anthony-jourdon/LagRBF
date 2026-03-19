#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "swarm/swarm_utils.h"
#include "readers/stagYY/stag_utils.h"

/* 
Create the DM Swarm to hold the points bearing the data to interpolate.
This swarm is special and its "lifecycle" is:

  1. Create the swarm as type DMSWARM_PIC
  2. Attach the sequential DMDA representing the rank-local background mesh
  First data loading phase:
    3. Load data into the swarm
    4. Migrate a) swarm points according to the PARALLEL DMDA decomposition => unique ownership of points => Use DMSWARM_MIGRATE_BASIC
               b) swarm points according to the SEQUENTIAL DMDA decomposition to duplicate the ghost points => Use custom migration function
    5. Create CellInfo and Partition structures

  Then during loop over time steps repeat steps 3. and 4.
*/
PetscErrorCode SwarmDataCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               swarm,das;
  PetscInt         dim,max_rank;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // Need to take care to manually change the migrate type
  PetscCall(DMSwarmSetCellDM(swarm, das));
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  if      (dim == 2) { max_rank = 4; }
  else if (dim == 3) { max_rank = 8; }
  else { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Dimension can only be 2 or 3. Found: %" PetscInt_FMT,dim); }

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"rbf",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"rank_to_go",max_rank,PETSC_INT));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  PetscCall(DMSetUp(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmDataAssignMigrateDuplicateGhostPoints(LagRBFCtx ctx, DM swarm)
{
  PartitionOfUnity pu;
  DM               das;
  PetscInt         np,dim,npoints,max_nranks;
  PetscInt         *rank_to_go;
  PetscReal        dms_min[3],dms_max[3];
  PetscReal        *all_dm_min = NULL, *all_dm_max = NULL;
  PetscReal        *pcoor;
  MPI_Comm         comm;
  PetscMPIInt      nranks,rank;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Get MPI info */
  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&nranks));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDALocal(pu,&das)); // rank-local overlapping sequential DMDAs
  PetscCall(DMGetDimension(das,&dim));

  /* Now we use the overlapping DMDA to duplicate points that belong to multiple ranks */
  PetscCall(DMGetLocalBoundingBox(das,dms_min,dms_max));
  /* Create a buffer to gather all the local bounding boxes */
  PetscCall(PetscMalloc1(3*nranks,&all_dm_min));
  PetscCall(PetscMalloc1(3*nranks,&all_dm_max));
  PetscCallMPI(MPI_Allgather(dms_min, 3, MPIU_REAL, all_dm_min, 3, MPIU_REAL, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allgather(dms_max, 3, MPIU_REAL, all_dm_max, 3, MPIU_REAL, PETSC_COMM_WORLD));
  max_nranks = (dim == 2) ? 4 : 8; // 4 in 2D, 8 in 3D
  /* Now check for each point if it is inside the local bounding box of each rank of the overlapping DMDA */
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  if (npoints < 0) { 
    DMSwarmSetLocalSizes(swarm,0,-1); 
    npoints = 0; 
  }
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmGetField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
  for (np=0; np<npoints; np++) {
    PetscInt  d,r,cnt;
    /* Initialize rank_to_go to -1 */
    for (r=0; r<max_nranks; r++) {
      rank_to_go[np*max_nranks + r] = -1;
    }
    /* Find all ranks the point belongs to */
    cnt = 0;
    for (r=0; r<nranks; r++) {
      PetscBool point_inside = PETSC_TRUE;
      for (d=0; d<dim; d++) {
        if (pcoor[dim*np+d] < all_dm_min[3*r + d]) point_inside = PETSC_FALSE;
        if (pcoor[dim*np+d] > all_dm_max[3*r + d]) point_inside = PETSC_FALSE;
      }
      if (point_inside) {
        rank_to_go[np*max_nranks + cnt] = r;
        cnt++;
      }
    }
    PetscCheck(cnt > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Point %" PetscInt_FMT " does not belong to any DMDA local bounding box.",np);
    //PetscCheck(cnt <= max_nranks,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Point %" PetscInt_FMT " belongs to more than %" PetscInt_FMT " ranks overlapping DMDA bounding boxes.",np,max_nranks);
  }
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmRestoreField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
  
  PetscCall(DMSwarmMigrate_Push_DuplicatePoints(swarm,0,npoints));
  
  PetscCall(PetscFree(all_dm_min));
  PetscCall(PetscFree(all_dm_max));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmDataLoad(LagRBFCtx ctx, DM sw)
{
  PetscFunctionBeginUser;
  /* For now there is only stag data, but the idea is to be able to choose between several type here */
  PetscCall(StagLoadDataToSwarmFromFiles(ctx,sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}