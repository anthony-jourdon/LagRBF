static const char help[] = "Test to interpolate and advect particles from StagYY output.\n\n";

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "io_utils.h"
#include "partition_of_unity.h"
#include "partition.h"
#include "swarm/swarm_output.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "readers/reader_utils.h"

/* petsc "private" headers */
#include <petsc/private/dmswarmimpl.h>
#include "../src/dm/impls/swarm/data_bucket.h"
#include "../src/dm/impls/swarm/data_ex.h"

static PetscErrorCode DMSwarmMigrate_Push_MultipleRanks(DM swarm, PetscInt point_start, PetscInt point_end)
{
  DM_Swarm    *sw = (DM_Swarm*)swarm->data;
  PetscInt    r,dim,max_nrank;
  MPI_Comm    comm;
  PetscMPIInt nranks,rank;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &nranks));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetDimension(swarm,&dim));
  /* Maximum number of ranks a point can belong to */
  max_nrank = (dim == 2) ? 4 : 8; // 4 in 2D, 8 in 3D

  for (r=0; r<max_nrank; r++) {
    DMSwarmDataEx de;
    PetscInt      np,n_points_recv,npoints;
    PetscInt      *rankval,*rank_to_go;
    size_t        sizeof_dmswarm_point;
    void          *point_buffer,*recv_points;
    PetscBool     debug = PETSC_FALSE;

    if (r >= nranks) break; // no need to proceed for ranks that do not exist
    /* Create the data exchanger structure */
    PetscCall(DMSwarmDataExCreate(comm, 0, &de));
    PetscCall(DMSwarmDataExTopologyInitialize(de));
    /* Get fields */
    PetscCall(DMSwarmGetField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval)); // rank of each point
    PetscCall(DMSwarmGetField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go)); // array of ranks to which each point should go
    /* Set ranks indinces to be sent on */
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      /* -1 means the point is not migrating */
      if (rank_to_set == -1) { continue; } 
      rankval[np] = rank_to_set;
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { PetscCall(DMSwarmDataExTopologyAddNeighbour(de, _rank)); }
    }
    PetscCall(DMSwarmDataExTopologyFinalize(de));
    /* Count how much points to be sent on each identified ranks */
    PetscCall(DMSwarmDataExInitializeSendCount(de));
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      if (rank_to_set == -1) { continue; }
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { PetscCall(DMSwarmDataExAddToSendCount(de, _rank, 1)); }
    }
    PetscCall(DMSwarmDataExFinalizeSendCount(de));
    PetscCall(DMSwarmDataBucketCreatePackedArray(sw->db, &sizeof_dmswarm_point, &point_buffer));
    /* Pack data to be sent */
    PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      if (rank_to_set == -1) { continue; }
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { 
        /* copy point into buffer */
        PetscCall(DMSwarmDataBucketFillPackedArray(sw->db, np, point_buffer));
        /* insert point buffer into DMSwarmDataExchanger */
        PetscCall(DMSwarmDataExPackData(de, _rank, 1, point_buffer));
      }
    }
    PetscCall(DMSwarmDataExPackFinalize(de));
    PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(DMSwarmRestoreField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
    /* Start the communication */
    PetscCall(DMSwarmDataExBegin(de));
    PetscCall(DMSwarmDataExEnd(de));
    PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
    /* Resize the DMSwarm data bucket to accommodate new points */
    PetscCall(DMSwarmDataBucketGetSizes(sw->db, &npoints, NULL, NULL));
    PetscCall(DMSwarmDataBucketSetSizes(sw->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
    for (np=0; np<n_points_recv; np++) {
      void *data_p = (void *)((char *)recv_points + np * sizeof_dmswarm_point);

      PetscCall(DMSwarmDataBucketInsertPackedArray(sw->db, npoints + np, data_p));
    }
    if (debug) PetscCall(DMSwarmDataExView(de));
    PetscCall(DMSwarmDataBucketDestroyPackedArray(sw->db, &point_buffer));
    PetscCall(DMSwarmDataExDestroy(de));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmMigrate_Push_MultipleRanks_vtest(DM swarm, PetscInt point_start, PetscInt point_end)
{
  DM_Swarm    *sw = (DM_Swarm*)swarm->data;
  PetscInt    r,dim,max_nrank,message_length;
  MPI_Comm    comm;
  PetscMPIInt nranks,rank;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &nranks));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetDimension(swarm,&dim));
  /* Maximum number of ranks a point can belong to */
  max_nrank = (dim == 2) ? 4 : 8; // 4 in 2D, 8 in 3D

  r = 0;
  message_length = 1;
  while (message_length > 0) {
    DMSwarmDataEx de;
    PetscInt      np,n_points_recv,npoints;
    PetscInt      *rankval,*rank_to_go;
    size_t        sizeof_dmswarm_point;
    void          *point_buffer,*recv_points;
    PetscBool     debug = PETSC_FALSE;
    
    /* Create the data exchanger structure */
    PetscCall(DMSwarmDataExCreate(comm, 0, &de));
    PetscCall(DMSwarmDataExTopologyInitialize(de));
    /* Get fields */
    PetscCall(DMSwarmGetField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval)); // rank of each point
    PetscCall(DMSwarmGetField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go)); // array of ranks to which each point should go
    /* Set ranks indinces to be sent on */
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      /* -1 means the point is not migrating */
      if (rank_to_set == -1) { continue; } 
      rankval[np] = rank_to_set;
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { 
        PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: np %d, rank_to_set %d\n",rank,(int)np,(int)rank_to_set);
        PetscCall(DMSwarmDataExTopologyAddNeighbour(de, _rank)); }
    }
    PetscCall(DMSwarmDataExTopologyFinalize(de));
    /* Count how much points to be sent on each identified ranks */
    PetscCall(DMSwarmDataExInitializeSendCount(de));
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      if (rank_to_set == -1) { continue; }
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { PetscCall(DMSwarmDataExAddToSendCount(de, _rank, 1)); }
    }
    PetscCall(DMSwarmDataExFinalizeSendCount(de));
    PetscCall(DMSwarmDataBucketCreatePackedArray(sw->db, &sizeof_dmswarm_point, &point_buffer));
    /* Pack data to be sent */
    PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
    PetscCallMPI(MPI_Allreduce(&de->send_message_length,&message_length,1,MPIU_INT,MPI_MAX,comm));
    for (np=point_start; np<point_end; np++) {
      PetscMPIInt _rank;
      PetscInt    rank_to_set = rank_to_go[np*max_nrank + r];
      if (rank_to_set == -1) { continue; }
      PetscCall(PetscMPIIntCast(rank_to_set, &_rank));
      if (_rank != rank) { 
        /* copy point into buffer */
        PetscCall(DMSwarmDataBucketFillPackedArray(sw->db, np, point_buffer));
        /* insert point buffer into DMSwarmDataExchanger */
        PetscCall(DMSwarmDataExPackData(de, _rank, 1, point_buffer));
      }
    }
    PetscCall(DMSwarmDataExPackFinalize(de));
    PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(DMSwarmRestoreField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
    /* Start the communication */
    PetscCall(DMSwarmDataExBegin(de));
    PetscCall(DMSwarmDataExEnd(de));
    PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
    /* Resize the DMSwarm data bucket to accommodate new points */
    PetscCall(DMSwarmDataBucketGetSizes(sw->db, &npoints, NULL, NULL));
    PetscCall(DMSwarmDataBucketSetSizes(sw->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
    for (np=0; np<n_points_recv; np++) {
      void *data_p = (void *)((char *)recv_points + np * sizeof_dmswarm_point);

      PetscCall(DMSwarmDataBucketInsertPackedArray(sw->db, npoints + np, data_p));
    }
    if (debug) PetscCall(DMSwarmDataExView(de));
    PetscCall(DMSwarmDataBucketDestroyPackedArray(sw->db, &point_buffer));
    PetscCall(DMSwarmDataExDestroy(de));
    r++;
    if (r >= max_nrank) message_length = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarmDataTest(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               das,dap,swarm;
  PetscInt         dim,max_rank;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PUGetDMDALocal(pu,&das));

  PetscCall(DMGetDimension(dap,&dim));
  /* Maximum number of neighbouring ranks */
  if      (dim == 2) { max_rank = 4; }
  else if (dim == 3) { max_rank = 8; }
  else { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Dimension can only be 2 or 3. Found: %" PetscInt_FMT,dim); }

  PetscCall(DMCreate(PETSC_COMM_WORLD,&swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_BASIC)); // Will use DMSwarmMigrate_Push_Basic
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm)); // this register by default the fields DMSwarmField_pid & DMSwarmField_rank

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"coords",dim,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"rank_to_go",max_rank,PETSC_INT32));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fidx",1,PETSC_INT64)); // index of the field in the output files

  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));

  { /* For debugging */
    MPI_Comm comm;
    int      rank;
    char     comm_name[MPI_MAX_OBJECT_NAME];

    PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(PetscSNPrintf(comm_name,MPI_MAX_OBJECT_NAME-1,"comm_%d",rank));
    PetscCallMPI(MPI_Comm_set_name(comm,comm_name));
  }


  *sw = swarm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmDataSetPoints_Rank0(LagRBFCtx ctx, DM swarm)
{
  PartitionOfUnity pu;
  //DM_Swarm         *sw = (DM_Swarm*)swarm->data;
  DM               das,dap;
  PetscInt         r,dim,d;
  PetscInt         np,npoints,max_nrank,np_init,np_new;
  PetscInt         *rankval,*rank_to_go;
  PetscInt64       *pidx;
  PetscReal        dms_min[3],dms_max[3],dmp_min[3],dmp_max[3];
  PetscReal        O[] = {0.25, 0.0, 0.0};
  PetscReal        L[] = {0.75, 2.0*PETSC_PI, PETSC_PI};
  PetscReal        *all_dm_min = NULL, *all_dm_max = NULL;
  int              nranks,rank;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)swarm),&nranks));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)swarm),&rank));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));      // parallel non-overlapping DMDA
  PetscCall(PUGetDMDALocal(pu,&das)); // rank-local overlapping sequential DMDAs
  PetscCall(DMGetDimension(das,&dim));

  PetscCall(DMGetBoundingBox(dap,dmp_min,dmp_max));
  PetscCall(DMGetLocalBoundingBox(das,dms_min,dms_max));
  /* Create a buffer to gather all the local bounding boxes */
  PetscCall(PetscMalloc1(3*nranks,&all_dm_min));
  PetscCall(PetscMalloc1(3*nranks,&all_dm_max));
  /* Gather on rank 0 all local bounding boxes */
  //PetscCallMPI(MPI_Gather(dms_min, 3, MPIU_REAL, all_dm_min, 3, MPIU_REAL, 0, PETSC_COMM_WORLD));
  //PetscCallMPI(MPI_Gather(dms_max, 3, MPIU_REAL, all_dm_max, 3, MPIU_REAL, 0, PETSC_COMM_WORLD));
  /* Other option: use MPI_Allgather to gather all local bounding boxes on all ranks */
  PetscCallMPI(MPI_Allgather(dms_min, 3, MPIU_REAL, all_dm_min, 3, MPIU_REAL, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allgather(dms_max, 3, MPIU_REAL, all_dm_max, 3, MPIU_REAL, PETSC_COMM_WORLD));

  /* Initialize the swarm on all ranks to size 0 */
  PetscCall(DMSwarmGetLocalSize(swarm,&np_init));
  if (np_init < 0) {
    np_init = 0;
    PetscCall(DMSwarmSetLocalSizes(swarm,0,-1));
  }
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Initial swarm size: %" PetscInt_FMT "\n",rank,np_init);
  max_nrank = 4; // for 2D
  /* 
    For testing purposes, only rank 0 create points and send them to the appropriate ranks
    But, in practice, if the data are into several files, we can have multiple ranks creating points
    Keep this in mind. 
  */
  if (rank == 0) {
    PetscReal  dphi;
    PetscReal  *pcoor;
    char       option_name[PETSC_MAX_PATH_LEN];
    const char *prefix;

    for (r=0; r<nranks; r++) {
      PetscPrintf(
        PETSC_COMM_WORLD,
        "Rank %d: DM local bounding box min: (%g,%g,%g) max: (%g,%g,%g)\n",
        r,all_dm_min[3*r+0],all_dm_min[3*r+1],all_dm_min[3*r+2],all_dm_max[3*r+0],all_dm_max[3*r+1],all_dm_max[3*r+2]
      );
    }

    PetscCall(DMGetOptionsPrefix(swarm,&prefix));
    PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%snpoints",prefix));
    npoints = 1000;
    PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&npoints,NULL));
    /* 
      For practical purposes, let us consider that 
      all the points are in the global DMDA because 
      the user should create a DMDA large enough to 
      contain all the data points
    */
    dphi = (L[1] - O[1]) / (PetscReal)npoints;
    PetscCall(DMSwarmGetLocalSize(swarm,&np_init));
    if (np_init == 0) {
      PetscCall(DMSwarmSetLocalSizes(swarm,npoints,-1));
    } else {
      PetscCall(DMSwarmAddNPoints(swarm,npoints)); // add npoints to the swarm (local)
    }
    PetscCall(DMSwarmGetLocalSize(swarm,&np_new)); // should be np_init + npoints

    PetscCall(DMSwarmGetField(swarm, "coords", NULL, NULL, (void **)&pcoor));
    PetscCall(DMSwarmGetField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
    PetscCall(DMSwarmGetField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
    //PetscCall(DMSwarmGetField(swarm, "pidx", NULL, NULL, (void **)&pidx));
    for (np=np_init; np<np_new; np++) {
      PetscInt  cnt;
      PetscReal u;
      PetscReal params[2];

      /* Initialize rank_to_go to -1 => is there a more efficient way? */
      for (d=0; d<max_nrank; d++) {
        rank_to_go[np*max_nrank + d] = -1;
      }

      /* Set coordinates */
      u = halton_sequence(np,2); // [0,1]
      /* Radius */
      params[0] = PetscSqrtReal( ( L[0]*L[0] - O[0]*O[0] )*u + O[0]*O[0] );
      /* Angle */
      params[1] = O[1] + np * dphi; 
      /* Cartesian coordinates */
      pcoor[2*np  ] = params[0] * PetscCosReal(params[1]);
      pcoor[2*np+1] = params[0] * PetscSinReal(params[1]);
      pidx[np] = np; // initial point index => ok if all points are created on rank 0 but not if points are created on multiple ranks
      PetscPrintf(PETSC_COMM_WORLD,"Point %d, pidx = %ld\n",(int)np,pidx[np]);
      /* Check on which processor this point should be sent (can be more than 1) */
      cnt = 0;
      for (r=0; r<nranks; r++) {
        PetscBool point_inside = PETSC_TRUE;
        /* Check if the point is in the local bounding box of rank r */
        for (d=0; d<dim; d++) {
          if (pcoor[2*np+d] < all_dm_min[3*r+d]) point_inside = PETSC_FALSE;
          if (pcoor[2*np+d] > all_dm_max[3*r+d]) point_inside = PETSC_FALSE;
        }
        if (point_inside) {
          rank_to_go[np*max_nrank + cnt] = r;
          cnt++;
        }
      }

      PetscPrintf(PETSC_COMM_WORLD,"Point %" PetscInt_FMT " with coordinates (%g,%g) will go to ranks: \n",np,pcoor[2*np],pcoor[2*np+1]);
      for (r=0; r<max_nrank; r++) {
        PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " ",rank_to_go[np*max_nrank + r]);
      }
      PetscPrintf(PETSC_COMM_WORLD,"\n");

      /* Ensure that at least one rank has been selected */
      PetscCheck(cnt > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Point %" PetscInt_FMT " with coordinates (%g,%g) does not belong to any DMDA local bounding box.",np,pcoor[2*np],pcoor[2*np+1]);
    }
    PetscCall(DMSwarmRestoreField(swarm, "coords", NULL, NULL, (void **)&pcoor));
    PetscCall(DMSwarmRestoreField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
    PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
    /* Here, all the points on rank 0 have registered the ranks where they will need to be sent, now we need to send them */
  }

  //PetscCall(DMSwarmDataBucketGetSizes(sw->db, &np_new, NULL, NULL));
  PetscCall(DMSwarmGetLocalSize(swarm,&np_new));
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Swarm size before migration: %" PetscInt_FMT "\n",rank,np_new);
  PetscCall(DMSwarmMigrate_Push_MultipleRanks(swarm,np_init,np_new));
  //PetscCall(DMSwarmDataBucketGetSizes(sw->db, &np_new, NULL, NULL));
  PetscCall(DMSwarmGetLocalSize(swarm,&np_new));
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Swarm size after migration: %" PetscInt_FMT "\n",rank,np_new);
  PetscCall(PetscFree(all_dm_min));
  PetscCall(PetscFree(all_dm_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmData_CheckPoints(DM swarm)
{
  PetscInt    np,npoints;
  PetscInt64  *pidx;
  PetscReal   *pcoor;
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));

  PetscCall(DMSwarmGetField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmGetField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  for (np=0; np<npoints; np++) {
    PetscPrintf(
      PETSC_COMM_SELF,
      "Rank[%d]: Point %" PetscInt_FMT " with pidx = %ld has coordinates (%g,%g)\n",
      rank, np, pidx[np], pcoor[2*np], pcoor[2*np+1]
    );
  }
  PetscCall(DMSwarmRestoreField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_GetFileList(PetscInt *nfiles, char ***file_list)
{
  PetscInt  f,nf;
  char      coor_dir[PETSC_MAX_PATH_LEN];
  char      **flist,**fbuffer;
  PetscBool flg;
  PetscFunctionBeginUser;
  
  PetscCall(PetscSNPrintf(coor_dir,PETSC_MAX_PATH_LEN-1,"./"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-coor_dir",coor_dir,PETSC_MAX_PATH_LEN,&flg));
  if (!flg) { PetscPrintf(PETSC_COMM_WORLD,"No coordinate directory specified, default: %s.\nChange it using -coor_dir <dir>\n",coor_dir); }

  /* Read coordinate files from coor_dir */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ncoor_files",&nf,&flg));
  if (!flg) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Need to specify number of coordinate files using -ncoor_files <nfiles>"); }
  PetscCall(PetscMalloc1(nf,&flist));
  PetscCall(PetscMalloc1(nf,&fbuffer));
  for (f=0; f<nf; f++) { 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&flist[f])); 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fbuffer[f]));
  }
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_files",fbuffer,&nf,&flg));
  for (f=0; f<nf; f++) {
    PetscCall(PetscSNPrintf(flist[f],PETSC_MAX_PATH_LEN-1,"%s/%s",coor_dir,fbuffer[f]));
    PetscFree(fbuffer[f]);
  }
  PetscCall(PetscFree(fbuffer));
  if (nfiles)    { *nfiles = nf; }
  if (file_list) { *file_list = flist; }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_SwarmAssignMigrate_Basic(LagRBFCtx ctx, DM swarm)
{
  PartitionOfUnity   pu;
  DMSwarmMigrateType migrate_type;
  DM                 dap;
  PetscInt           np,npoints,dim;
  PetscInt           *rankval;
  PetscReal          dmp_min[3],dmp_max[3];
  PetscReal          *pcoor;
  PetscReal          *all_dmp_min = NULL, *all_dmp_max = NULL;
  MPI_Comm           comm;
  PetscMPIInt        nranks,rank;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap)); // parallel non-overlapping DMDA
  PetscCall(DMGetDimension(dap,&dim));

  /* Get MPI info */
  PetscCall(PetscObjectGetComm((PetscObject)dap,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&nranks));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  /* First we use the non-overlapping DMDA to distribute the swarm points UNIQUELY */
  /* Get rank-local bounding box */
  PetscCall(DMGetLocalBoundingBox(dap,dmp_min,dmp_max));
  PetscCall(PetscMalloc1(3*nranks,&all_dmp_min));
  PetscCall(PetscMalloc1(3*nranks,&all_dmp_max));
  /* Gather the information from all ranks */
  PetscCallMPI(MPI_Allgather(dmp_min, 3, MPIU_REAL, all_dmp_min, 3, MPIU_REAL, comm));
  PetscCallMPI(MPI_Allgather(dmp_max, 3, MPIU_REAL, all_dmp_max, 3, MPIU_REAL, comm));
  /* Now check for each point if it is inside the local bounding box of the non-overlapping DMDA */
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmGetField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  for (np=0; np<npoints; np++) {
    PetscInt    d;
    PetscMPIInt nr;
    /* Find which rank the point belongs to */
    rankval[np] = -1; // initialize to invalid rank
    for (nr=0; nr<nranks; nr++) {
      PetscBool point_inside = PETSC_TRUE;
      for (d=0; d<dim; d++) {
        if (pcoor[dim*np+d] < all_dmp_min[3*nr + d]) point_inside = PETSC_FALSE;
        if (pcoor[dim*np+d] > all_dmp_max[3*nr + d]) point_inside = PETSC_FALSE;
      }
      if (point_inside) {
        rankval[np] = nr;
        break;
      }
    }
    PetscCheck(rankval[np] != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Point %" PetscInt_FMT ", with coordinates (%g,%g) does not belong to any DMDA local bounding box.",np,pcoor[dim*np],pcoor[dim*np+1]);
  }
  PetscCall(DMSwarmRestoreField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  /* 
    Now that each point has a unique rank assigned, 
    we can use the basic migration and remove the points 
    that do not belong to this rank 
  */
  PetscCall(DMSwarmGetMigrateType(swarm,&migrate_type));
  /* Ensure we use the basic migration */
  if (migrate_type != DMSWARM_MIGRATE_BASIC) { PetscCall(DMSwarmSetMigrateType(swarm,DMSWARM_MIGRATE_BASIC)); }
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  /* Now assign a unique index to these points */
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm,DMSwarmField_pid,0,npoints));

  PetscCall(PetscFree(all_dmp_min));
  PetscCall(PetscFree(all_dmp_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_SwarmAssignDuplicatePoints(LagRBFCtx ctx, DM swarm)
{
  PartitionOfUnity pu;
  DM               das;
  PetscInt64       *pidx;
  PetscInt         np,npoints,dim,max_nranks,np_new;
  PetscInt         *rank_to_go;
  PetscReal        dms_min[3],dms_max[3];
  PetscReal        *all_dm_min = NULL, *all_dm_max = NULL;
  PetscReal        *pcoor;
  MPI_Comm         comm;
  PetscMPIInt      nranks,rank;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&nranks));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDALocal(pu,&das)); // rank-local overlapping sequential DMDAs
  DMGetDimension(das,&dim);
  /* Now we use the overlapping DMDA to duplicate points that belong to multiple ranks */
  PetscCall(DMGetLocalBoundingBox(das,dms_min,dms_max));
  /* Create a buffer to gather all the local bounding boxes */
  PetscCall(PetscMalloc1(3*nranks,&all_dm_min));
  PetscCall(PetscMalloc1(3*nranks,&all_dm_max));
  PetscCallMPI(MPI_Allgather(dms_min, 3, MPIU_REAL, all_dm_min, 3, MPIU_REAL, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allgather(dms_max, 3, MPIU_REAL, all_dm_max, 3, MPIU_REAL, PETSC_COMM_WORLD));

  /* Now check for each point if it is inside the local bounding box of each rank of the overlapping DMDA */
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmGetField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
  PetscCall(DMSwarmGetField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  max_nranks = (dim == 2) ? 4 : 8; // 4 in 2D, 8 in 3D
  for (np=0; np<npoints; np++) {
    PetscInt  d,r,cnt;
    /* Initialize rank_to_go to -1 */
    for (r=0; r<max_nranks; r++) {
      rank_to_go[np*max_nranks + r] = -1;
    }
    /* Find all ranks the point belongs to */
    cnt = 0;
    for (r=0; r<max_nranks; r++) {
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
    PetscCheck(cnt > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Point %" PetscInt_FMT " with coordinates (%g,%g) does not belong to any DMDA local bounding box.",np,pcoor[dim*np],pcoor[dim*np+1]);
    PetscPrintf(
      PETSC_COMM_SELF,
      "Rank %d: Point %" PetscInt_FMT ", pidx %ld, with coordinates (%g,%g) will go to ranks: [ %d, %d, %d, %d ] \n",
      rank,np,pidx[np],pcoor[dim*np],pcoor[dim*np+1],(int)rank_to_go[np*max_nranks],(int)rank_to_go[np*max_nranks + 1],(int)rank_to_go[np*max_nranks + 2],(int)rank_to_go[np*max_nranks + 3]
    );
  }
  PetscCall(DMSwarmRestoreField(swarm, "coords", NULL, NULL, (void **)&pcoor));
  PetscCall(DMSwarmRestoreField(swarm, "rank_to_go", NULL, NULL, (void **)&rank_to_go));
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Swarm size before migration: %" PetscInt_FMT "\n",rank,npoints);
  PetscCall(DMSwarmMigrate_Push_MultipleRanks(swarm,0,npoints));
  PetscCall(DMSwarmGetLocalSize(swarm,&np_new));
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Swarm size after migration: %" PetscInt_FMT "\n",rank,np_new);
  
  PetscCall(PetscFree(all_dm_min));
  PetscCall(PetscFree(all_dm_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_InitializeSwarmFromFiles(DM swarm, char **file_list, PetscInt fstart, PetscInt fend)
{
  PetscInt    f,np_init;
  PetscReal   *pcoor;
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(DMSwarmGetLocalSize(swarm,&np_init));
  if (np_init < 0) {
    np_init = 0;
    PetscCall(DMSwarmSetLocalSizes(swarm,0,-1));
  }

  for (f=fstart; f<fend; f++) {
    FILE     *fp;
    PetscInt l,np_current,np_new;
    char     line[PETSC_MAX_PATH_LEN];
    
    /* Step 1: count how much points to add to the swarm */
    fp = fopen(file_list[f],"r");
    if (!fp) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Rank %d: Could not open file %s\n",rank,file_list[f]); }
    /* determine how much lines there are in the file in our toy example lines = points */
    l = 0;
    while (fgets(line,PETSC_MAX_PATH_LEN,fp)) { l++; }
    PetscCall(DMSwarmGetLocalSize(swarm,&np_current));
    PetscCall(DMSwarmAddNPoints(swarm,l)); // add l points to the swarm (local)
    PetscCall(DMSwarmGetLocalSize(swarm,&np_new));
    /* Go back to start of the file */
    rewind(fp);
    /* Step 2: read points from the file and assign to the swarm */
    PetscCall(DMSwarmGetField(swarm, "coords", NULL, NULL, (void **)&pcoor));
    l = 0;
    while (fgets(line,PETSC_MAX_PATH_LEN,fp)) {
      PetscReal x,y;
      sscanf(line,"%lf %lf",&x,&y);
      pcoor[2*(np_current + l)  ] = x;
      pcoor[2*(np_current + l)+1] = y;
      PetscPrintf(
        PETSC_COMM_SELF,
        "Rank %d, File %d, point local file idx %d, point local swarm idx %d, coor (%g,%g)\n",
        rank,f,l,np_current+l,x,y
      );
      l++;
    }
    PetscCall(DMSwarmRestoreField(swarm, "coords", NULL, NULL, (void **)&pcoor));
    fclose(fp);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_ReadFiles(DM swarm)
{
  char        coor_dir[PETSC_MAX_PATH_LEN];
  char        **file_list,**file_buffer;
  PetscInt    f,nfiles,idx_start,idx_end,nlocal_files,np_init;
  PetscInt64  rank_lines;
  PetscBool   flg;
  PetscMPIInt rank;
  MPI_Comm    comm;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));

  PetscCall(PetscSNPrintf(coor_dir,PETSC_MAX_PATH_LEN-1,"./"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-coor_dir",coor_dir,PETSC_MAX_PATH_LEN,&flg));
  if (!flg) {
    PetscPrintf(PETSC_COMM_WORLD,"No coordinate directory specified, default: %s.\nChange it using -coor_dir <dir>\n",coor_dir);
  }
  /* Read coordinate files from coor_dir */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ncoor_files",&nfiles,&flg));
  if (!flg) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Need to specify number of coordinate files using -ncoor_files <nfiles>"); }
  PetscCall(PetscMalloc1(nfiles,&file_list));
  PetscCall(PetscMalloc1(nfiles,&file_buffer));
  for (f=0; f<nfiles; f++) { 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&file_list[f])); 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&file_buffer[f]));
  }
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_files",file_buffer,&nfiles,&flg));
  for (f=0; f<nfiles; f++) {
    PetscCall(PetscSNPrintf(file_list[f],PETSC_MAX_PATH_LEN-1,"%s/%s",coor_dir,file_buffer[f]));
    PetscPrintf(PETSC_COMM_WORLD,"File %d: %s\n",f,file_list[f]);
    PetscFree(file_buffer[f]);
  }
  PetscFree(file_buffer);

  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  ReadersGetRankDistribution(comm,nfiles,&idx_start,&idx_end,&nlocal_files);
  
  /* Initialize the swarm on all ranks to size 0 */
  PetscCall(DMSwarmGetLocalSize(swarm,&np_init));
  if (np_init < 0) {
    np_init = 0;
    PetscCall(DMSwarmSetLocalSizes(swarm,0,-1));
  }

  rank_lines = 0;
  for (f=idx_start; f<idx_end; f++) {
    FILE *fp;
    PetscInt l,np_current,np_new;
    char line[PETSC_MAX_PATH_LEN];

    PetscPrintf(PETSC_COMM_SELF,"Rank %d: Reading file %s (file %d of %d)\n",rank,file_list[f],f+1,nfiles);
    
    fp = fopen(file_list[f],"r");
    if (!fp) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Rank %d: Could not open file %s for reading\n",rank,file_list[f]); }
    /* determine how much lines there are in the file in this case lines = points */
    l = 0;
    while (fgets(line,PETSC_MAX_PATH_LEN,fp)) { l++; }
    fclose(fp);
    PetscCall(DMSwarmGetLocalSize(swarm,&np_current));
    if (np_current == 0) { PetscCall(DMSwarmSetLocalSizes(swarm,l,-1)); } // set initial size
    else                 { PetscCall(DMSwarmAddNPoints(swarm,l)); }       // add l points to the swarm (local)
    PetscCall(DMSwarmGetLocalSize(swarm,&np_new));
    

    /* assign a unique index from the lines to the swarm */
    //PetscCall(SwarmFieldAssignUniqueIndex(swarm,"fidx",rank_lines,rank_lines + l));
    rank_lines += l;


#if 0
    PetscPrintf(PETSC_COMM_SELF,"Rank %d: File %s has %d lines\n",rank,file_list[f],l);
    /* Read coordinates from file */
    l = 0;
    while (fgets(line,PETSC_MAX_PATH_LEN,fp) != NULL) {
      PetscReal x,y;
      sscanf(line,"%lf %lf",&x,&y);
      PetscPrintf(PETSC_COMM_SELF,"Rank %d: line %d: Read coordinates (%g,%g)\n",rank,(int)l,x,y);
      l++;
    }
#endif
    //fclose(fp);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test(void)
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  DM               swarm_data;
  PetscInt         global_nfiles,local_nfiles,fstart,fend;
  PetscBool        exists;
  MPI_Comm         comm = PETSC_COMM_WORLD;
  char             **global_flist;
  PetscFunctionBeginUser;
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));

  PetscCall(Test_GetFileList(&global_nfiles,&global_flist));
  ReadersGetRankDistribution(comm,global_nfiles,&fstart,&fend,&local_nfiles);
  PetscCall(CreateSwarmDataTest(ctx, &swarm_data));
  PetscCall(Test_InitializeSwarmFromFiles(swarm_data,global_flist,fstart,fend));
  /* Now that swarm has been populated start the migration */
  PetscCall(Test_SwarmAssignMigrate_Basic(ctx,swarm_data));
  PetscCall(Test_SwarmAssignDuplicatePoints(ctx,swarm_data));

  //PetscCall(Test_ReadFiles(swarm_data));
  //PetscCall(SwarmDataSetPoints_Rank0(ctx, swarm_data));
  PetscCall(SwarmData_CheckPoints(swarm_data));

  PetscCall(PartitionOfUnityMeshView(pu));
  PetscCall(PartitionOfUnityMeshView_Seq(pu));


  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,help));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}