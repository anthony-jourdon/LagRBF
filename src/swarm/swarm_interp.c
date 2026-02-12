#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "advection/advection_ctx.h"
#include "kernels/kernels_rbf.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "cell_info.h"
#include "io_utils.h" 

PetscLogEvent LAGRBF_SwarmInterpSetWeights_Scalar;

PetscErrorCode SwarmInterpCreateAdvectionFields(LagRBFCtx ctx, DM swarm)
{
  AdvectionCtx adv_ctx;
  PetscInt     dim,order;
  const char   *velocity_name;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ Advection activated ]]: creating advection fields\n");
  
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetIntegrationOrder(adv_ctx,&order));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  /* Velocity field interpolated */
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering velocity field \"%s\" of block size: %" PetscInt_FMT "\n",velocity_name,dim);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,velocity_name,dim,PETSC_REAL));
  /* If the scheme is higher than first order, we need to store the initial coordinates and the velocity at each integration step */
  if (adv_ctx->scheme != RK1 || adv_ctx->scheme != RK2) {
    PetscInt b;
    /* Store initial coordinates before advection */
    PetscPrintf(PETSC_COMM_WORLD,"\tRegistering initial coordinates field \"coor_init\" of block size: %" PetscInt_FMT "\n",dim);
    PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"coor_init",dim,PETSC_REAL));
    for (b=0; b<order; b++) {
      char v_adv_name[PETSC_MAX_PATH_LEN];
      /* Register Velocity at integration step */
      PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",b));
      PetscPrintf(PETSC_COMM_WORLD,"\tRegistering integration step velocity field \"%s\" of block size: %" PetscInt_FMT "\n",v_adv_name,dim);
      PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,v_adv_name,dim,PETSC_REAL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               dap,swarm;
  PetscInt         dim,nel,nen;
  const PetscInt   *elidx;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  /* Create interpolation points */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(DMGetDimension(dap,&dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dap), &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, dap));
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  /* Weights for interpolation */
  PetscCall(DMDAGetElements(dap,&nel,&nen,&elidx));
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering \"weights\" real field of block size: %" PetscInt_FMT "\n",nen);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",nen,PETSC_REAL));
  PetscCall(DMDARestoreElements(dap,&nel,&nen,&elidx));
  /* Fields to interpolate, also contain the velocity even in advection mode */
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering \"fields\" real field of block size: %" PetscInt_FMT "\n",ctx->n_ifields);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  /* If advection is required, special fields are added */
  if (ctx->advection) {
    PetscCall(SwarmInterpCreateAdvectionFields(ctx, swarm));
  }
  // May need to add another set of fields that need to be advected but not interpolated
  // PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pfields",ctx->n_pfields,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));
  /* Initialize with no particles */
  PetscCall(DMSwarmSetLocalSizes(swarm,0,-1));

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
    PetscCall(DMSwarmSetPointsSphericalCoordinates(sw,O,L,m,mode,PETSC_TRUE));
  } else {
    PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,mode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm)
{
  DMSwarmCellDM celldm;
  Vec           coor;
  PetscInt      bs,size,npoints;
  PetscReal     *pcoords,*coords;
  const char    **coordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Load coordinates from file */
  PetscCall(VecCreate(comm, &coor));
  PetscCall(VecLoadFromFile(coor, filename));
  PetscCall(VecGetBlockSize(coor,&bs));
  PetscCall(VecGetSize(coor,&size));
  npoints = (PetscInt)(size / bs);
  PetscPrintf(PETSC_COMM_WORLD,"  Loaded %" PetscInt_FMT " points of block size %" PetscInt_FMT " from file: %s\n",npoints,bs,filename);
  /* Set swarm size */
  PetscCall(DMSwarmSetLocalSizes(sw,npoints,-1));
  /* Get swarm coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Copy coordinates from vector to swarm */
  PetscCall(VecGetArray(coor,&coords));
  PetscCall(PetscArraycpy(pcoords,coords,size));
  PetscCall(VecRestoreArray(coor,&coords));
  /* Clean up */
  PetscCall(VecDestroy(&coor));
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm)
{
  PetscLayout    map;
  IS             pid;
  PetscInt       np,N,npoints;
  const PetscInt *is_pidx;
  PetscInt64     *pidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Load the point indices stored as Index Set (IS) */
  PetscCall(ISCreate(comm, &pid));
  PetscCall(ISSetType(pid,ISGENERAL));
  PetscCall(ISLoadFromFile(pid, filename));
  /* Check that IS is the same size than the swarm */
  PetscCall(ISGetLayout(pid, &map));
  PetscCall(PetscLayoutGetSize(map, &N));
  PetscCall(DMSwarmGetLocalSize(sw, &npoints));
  PetscCheck(N == npoints, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "[[ %s ]] Loaded point index size %" PetscInt_FMT " does not match swarm size %" PetscInt_FMT "", PETSC_FUNCTION_NAME, N, npoints);
  /* Copy point indices into the swarm */
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  PetscCall(ISGetIndices(pid, &is_pidx));
  for (np=0; np<npoints; np++) { pidx[np] = (PetscInt64)is_pidx[np]; }
  PetscCall(ISRestoreIndices(pid, &is_pidx));
  /* Clean up */
  PetscCall(ISDestroy(&pid));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscErrorCode SwarmInterpLoadFromFile(LagRBFCtx ctx, DM sw, const char *dirname, PetscBool load_pidx)
  
  Load the swarm coordinates and point indices from files located in 'dirname'.
  The files are expected to be named according to the convention:
    <stepprefix>.<dm_name>_<coord_field>.pbvec  for coordinates
    <stepprefix>.<dm_name>_pid.pbis            for point indices
  where <stepprefix> is derived from the current step in the LagRBFCtx context and potentially appended with "_forward" or "_backward" if advection is active.
  The point loading is only performed on rank 0, followed by a broadcast to other ranks using a basic migration strategy.
  The switch to a basic migration strategy is necessary because DMSWARM_MIGRATE_DMCELLNSCATTER only uses neighbor ranks for point migration,
  which can result in incomplete data distribution.
  After broadcasting, the migrate type is reset to DMSWARM_MIGRATE_DMCELLNSCATTER and cell IDs are assigned using the DMSwarmCellDM.
*/
PetscErrorCode SwarmInterpLoadFromFile(LagRBFCtx ctx, DM sw, const char *dirname, PetscBool load_pidx)
{
  DMSwarmCellDM celldm;
  PetscInt step;
  char filename[PETSC_MAX_PATH_LEN],stepprefix[PETSC_MAX_PATH_LEN];
  const char *dm_name,**coordFields;
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  PetscCall(LagRBFGetStep(ctx,&step));
  PetscCall(PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%06" PetscInt_FMT "",step));
  if (ctx->forward)  { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_forward", step)); }
  if (ctx->backward) { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_backward", step)); }
  
  /* Load coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscSNPrintf(filename, PETSC_MAX_PATH_LEN-1, "%s/%s.%s_%s.pbvec", dirname, stepprefix, dm_name, coordFields[0]);
  if (rank == 0) {
    PetscCall(SwarmInterpLoadCoordinatesFromFile(sw, filename, PETSC_COMM_SELF));
    /* Load point indices */
    if (load_pidx) {
      PetscSNPrintf(filename, PETSC_MAX_PATH_LEN-1, "%s/%s.%s_%s.pbis", dirname, stepprefix, dm_name, DMSwarmField_pid);
      PetscCall(SwarmInterpLoadPidxFromFile(sw, filename, PETSC_COMM_SELF));
    }
  }
  /* Ensure that all ranks have reached this point before proceeding */
  PetscCall(PetscBarrier((PetscObject)NULL));
  /* Send to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx, sw));
  /* Set back the migrate type to DMCELLNSCATTER */
  PetscCall(DMSwarmSetMigrateType(sw,DMSWARM_MIGRATE_DMCELLNSCATTER));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  To compute the weights in theory we would need to compute at each point:
    w_j = phi(||x - c_j||) / (sum_{k=1}^{N} phi(||x - c_k||)) 
  where:
    - w_j is the weight for partition j
    - phi is the RBF kernel function
    - x is the coordinate of the point
    - c_j is the center of partition j
    - N is the total number of partitions
  But in practice, because the kernel has compact support i.e., 
  phi(r) = 0 for r > R (R is the support radius),
  we only need to compute the weights for the partitions 
  whose centers are within the support radius of the point.
  Except if one discretizes the domain agressively, we should only need the nodes in the local DMDA
  and thus no communication should be needed.
*/
static PetscErrorCode SwarmInterpSetWeights_Scalar(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  DMSwarmCellDM    celldm;
  Partition        *partitions;
  DM               das,cdas;
  Vec              coords;
  PetscReal        epsilon;
  PetscReal        *pcoords,*weights,*LA_coords;
  PetscInt         k,np,n,dim,npart,npoints,nn,puidx,nel,nen;
  PetscInt         *cellid,*gtol_elidx;
  const PetscInt   *elidx;
  const char       **coordFields,*cellFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_Scalar,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Get data structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Get the rank-local DMDA (overlapping between processes) */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  /* Get partitions centre coordinates (DMDA's nodes) */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  PetscCall(DMGetCoordinates(das,&coords));
  PetscCall(VecGetArray(coords,&LA_coords));
  /* Get the number of nodes in the sequential DMDA */
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Get the parallel DMDA to sequential DMDA element map */
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Cell id */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellFieldName));
  PetscCall(DMSwarmGetField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCall(PUGetEpsilon(pu,&epsilon));
  for (np=0; np<npoints; np++) {
    PetscInt  eidx,eidx_local,nidx_e;
    PetscReal sum_w = 0.0;

    /* Cell index of the particle */
    eidx       = cellid[np];       // parallel DMDA
    eidx_local = gtol_elidx[eidx]; // local sequential DMDA
    /* Loop over the local sequential DMDA partitions */
    for (n=0; n<npart; n++) {
      Partition p = partitions[n]; // current partition
      PetscInt  d;
      PetscReal dist,weight;
      PetscReal *pu_centre;
      
      PetscCall(PartitionGetIndex(p,&puidx));
      pu_centre = &LA_coords[dim*puidx];
      dist = 0.0;
      for (d=0; d<dim; d++) { 
        dist += (pcoords[dim*np + d] - pu_centre[d]) * (pcoords[dim*np + d] - pu_centre[d]); 
      }
      dist   = PetscSqrtReal(dist);
      /* Compute the weight */
      weight = KernelWendlandC2(dist,epsilon);
      sum_w += weight;

      /* Store the weight only for the nodes connected to this element */
      for (k=0; k<nen; k++) {
        nidx_e = elidx[nen*eidx_local + k];
        if (nidx_e == puidx) {
          weights[np*nen + k] = weight;
          break;
        }
      }
    }
    /* Normalize weights */
    for (k=0; k<nen; k++) {
      nidx_e = elidx[nen*eidx_local + k];
      if (sum_w > 0.0) {
        /* Normalize the weights */
        weights[np*nen + k] /= sum_w;
      } else {
        /* If the sum of weights is zero, set the weight to zero */
        weights[np*nen + k] = 0.0;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coords,&LA_coords));

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_Scalar,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw)
{
  PetscFunctionBeginUser;
  /* No performance gain with AVX512 */
#if 0//defined(__AVX512F__)
  PetscCall(SwarmInterpSetWeights_AVX512(ctx,sw));
  //PetscCall(SwarmInterpSetWeights_AVX512_PointVectorized(ctx,sw));
#elif defined(__AVX__)
  PetscCall(SwarmInterpSetWeights_AVX256(ctx,sw));
  //PetscCall(SwarmInterpSetWeights_AVX256_PointVectorized(ctx,sw));
#else
  PetscCall(SwarmInterpSetWeights_Scalar(ctx,sw));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx)
{
  PetscInt      p,npoints,bs;
  PetscInt64    max_local,max,np_local,np_global;
  PetscInt64    *pidx;
  PetscDataType dtype;
  PetscMPIInt   rank;
  MPI_Comm      comm;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PetscObjectGetComm((PetscObject)sw,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, &bs, &dtype, (void **)&pidx));
  PetscPrintf(PETSC_COMM_WORLD,"pidx data type: %s, blocksize: %" PetscInt_FMT "\n",PetscDataTypes[dtype],bs);
  PetscCheck(dtype == PETSC_INT64, comm, PETSC_ERR_ARG_WRONGSTATE, "Field \"pidx\" has wrong data type. Expected PETSC_INT64, got %s", PetscDataTypes[dtype]);
  PetscCheck(bs == 1, comm, PETSC_ERR_ARG_WRONGSTATE, "Field \"pidx\" has wrong blocksize. Expected 1, got %" PetscInt_FMT "", bs);
  /* find max pid presently in the system */
  max_local = 0;
  for (p=0; p<npoints; p++) {
    PetscInt64 mp_idx = pidx[p]; 
    if ( mp_idx > max_local ) { max_local = mp_idx; }
  }
  PetscCallMPI(MPI_Allreduce(&max_local,&max,1,MPIU_INT64,MPI_MAX,comm));
  //PetscPrintf(comm,"[[ %s ]] : max_pid = %" PetscInt64_FMT "\n",PETSC_FUNCTION_NAME,max);
  max += 1;

  /* give particles a unique identifier */
  np_local = (PetscInt64)(end_idx - start_idx);
  PetscCallMPI(MPI_Scan(&np_local,&np_global,1,MPIU_INT64,MPI_SUM,comm));
  //printf("rank %d : np_local = %" PetscInt64_FMT ", np_global = %" PetscInt64_FMT "\n",(int)rank,np_local,np_global);
  for (p=start_idx; p<end_idx; p++) {
    pidx[p] = max + (np_global-np_local) + (PetscInt64)(p-start_idx);
    //printf("assigning %" PetscInt_FMT " -> pid = %" PetscInt64_FMT " \n", p, pidx[p] );
  }
  PetscCall(DMSwarmRestoreField(sw,DMSwarmField_pid,&bs,&dtype,(void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw)
{
  DM       swarm;
  //PetscInt npoints;
  PetscFunctionBeginUser;

  PetscCall(SwarmInterpCreate(ctx,&swarm));
  if (ctx->restart) {
    //PetscCall(SwarmInterpLoadCoordinates(pu,swarm));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Restart not implemented yet for Swarm Interpolation");
  } else {
    PetscCall(SwarmInterpSetCoordinatesFromOptions(ctx->pu,swarm,INSERT_VALUES));
  }
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  /* We may want to do this later */
  //PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  //PetscCall(SwarmInterpAssignUniquePointIndex(swarm,0,npoints));
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