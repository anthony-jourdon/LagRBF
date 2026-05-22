#include <petsc.h>
#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "io_utils.h"
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "swarm/swarm_output.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_load.h"
#include "readers/reader_utils.h"
#include "readers/stagYY/stag_utils.h"
#include "readers/stagYY/binary_loader.h"
#include "interpolation.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "cell_info.h"
#include "time_loop.h"

static PetscErrorCode DebugOutputDataSurface(LagRBFCtx ctx, DM swarm_data)
{
  DMSwarmCellDM celldm;
  PetscInt      np,npoints,cnt,dim;
  PetscInt     *rm_idx;
  PetscReal     rmin,rmax,tol;
  PetscReal    *pcoords;
  const char  **coordFields;
  const char   *output_fields[] = {"fields"};
  PetscFunctionBeginUser;
  
  PetscCall(DMGetDimension(swarm_data,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm_data,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm_data,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(swarm_data, coordFields[0], NULL, NULL, (void **)&pcoords));
  PetscCall(PetscMalloc1(npoints,&rm_idx));
  tol = 1e-6;
  rmin = 2.19*2.19;
  rmax = 2.2*2.2;
  cnt = 0;
  for (np=0; np<npoints; np++) {
    PetscInt  d;
    PetscReal radius2 = 0.0;
    for (d=0; d<dim; d++) {
      radius2 += pcoords[dim*np + d] * pcoords[dim*np + d];
    }
    if (radius2 > rmax+tol || radius2 < rmin-tol) {
      rm_idx[cnt++] = np;
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_data, coordFields[0], NULL, NULL, (void **)&pcoords));
  if (cnt > 0) {
    PetscCall(DMSwarmRemoveNPointsFromIndex(swarm_data, cnt, rm_idx));
  }
  PetscCall(PetscFree(rm_idx));
  PetscCall(SwarmOutput(ctx,swarm_data,1,output_fields,PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KeepOnlyContinents(
  LagRBFCtx ctx,
  DM        swarm_interp
)
{
  PetscInt   np,npoints_i,bs,cnt;
  PetscInt  *rm_idx;
  PetscReal *fields;
  PetscMPIInt rank;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(PetscMalloc1(npoints_i,&rm_idx));
  
  PetscCall(DMSwarmGetField(swarm_interp, "fields", &bs, NULL, (void**)&fields));
  cnt = 0;
  for (np=0; np<npoints_i; np++) {
    PetscInt region = 0;
    if (fields[bs*np    ] > 0.1) { region = 1; } // mark as archean
    if (fields[bs*np + 1] > 0.1) { region = 2; } // mark as proterozoic
    if (region == 0) {
      rm_idx[cnt++] = np; // mark point for removal if it does not belong to any continent
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_interp, "fields", &bs, NULL, (void**)&fields));
  if (cnt > 0) {
    /* Remove points */
    PetscCall(DMSwarmRemoveNPointsFromIndex(swarm_interp, cnt, rm_idx));
  }
  PetscCall(PetscFree(rm_idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetContinentMarkers(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  step
)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  PetscInt         f,nfiles;
  PetscMPIInt      rank;
  char             **filenames;
  PetscBool        *is_scalar;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  
  ctx->forward  = PETSC_FALSE;
  ctx->backward = PETSC_FALSE;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  /* Insert new points */
  PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));

  PetscCall(StagGetFilelistFromOptions(step,&nfiles,&filenames,&is_scalar));
  for (f=0; f<nfiles; f++) {
    PetscPrintf(PETSC_COMM_WORLD,"  File[%'" PetscInt_FMT "]: %s\n",f,filenames[f]);
  }
  /* Read data only on rank 0 */
  if (rank == 0) {
    PetscCall(StagBinSwarmSetFromFiles(swarm_data,filenames,nfiles,is_scalar));
  }
  PetscCall(StagRestoreFilelist(nfiles, &filenames, &is_scalar));
  /* Ensure that rank 0 has finished reading data */
  PetscCall(PetscBarrier((PetscObject)NULL));

  /* Send data to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx,swarm_data));

  //PetscCall(DebugOutputDataSurface(ctx,swarm_data)); // DEBUG

  /* Duplicate ghost points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,swarm_data));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(swarm_data));
  /* Set up CellInfo data structure */
  PetscCall(CellInfoSetUp(pu,ci,swarm_data));
  /* Interpolation */
  PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));

  /* Now we keep only the points with a continent value */
  PetscCall(KeepOnlyContinents(ctx,swarm_interp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Init()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscInt         nsteps,npoints_i;
  PetscInt         *steps;
  PetscReal        *time_steps;
  PetscBool        exists;
  const char      *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  
  /* Create the output directory */
  PetscCall(PetscTestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->output_dir); }

  /* Create data structures */
  ctx->mtype = STAGYY; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Create the swarm to hold data */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  
  PetscCall(GetTimeStepsFromOptions(&nsteps,&steps,&time_steps));
  PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %" PetscInt_FMT "\n",nsteps);

  /* We generate a swarm that only contains the continents */
  PetscCall(SetContinentMarkers(ctx,swarm_data,swarm_interp,steps[0]));

  /* Now that we have all our points we assign them a unique index */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  /* Output the initial points */
  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));

  /* Clean up */
  PetscCall(RestoreTimeSteps(&steps,&time_steps));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  Load an swarm from a previously saved state but consider it as a new initial condition
  Not a restart.
*/
static PetscErrorCode LoadInit(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  nsteps,
  PetscInt  steps[],
  PetscReal time_steps[]
)
{
  PartitionOfUnity pu;
  AdvectionScheme  adv_scheme;
  CellInfo         *ci;
  PetscMPIInt      rank;
  PetscInt  load_step,nfields,nn,s,restart_idx;
  PetscBool found,exists;
  char      option_prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN],restart_dir[PETSC_MAX_PATH_LEN];
  char      **fields_name = NULL;
  const char *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;
  
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Load swarm */
  PetscCall(PetscSNPrintf(option_prefix, PETSC_MAX_PATH_LEN-1, "load_"));
  /* Directory we load from */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sdirectory", option_prefix));
  PetscCall(PetscOptionsGetString(NULL,NULL,option_name,restart_dir,PETSC_MAX_PATH_LEN-1,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
  PetscCall(PetscTestDirectory(restart_dir,'r',&exists));
  PetscCheck(exists,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",restart_dir);
  
  /* Check if we load forward, backward or neither */
  ctx->forward  = PETSC_FALSE;
  ctx->backward = PETSC_FALSE;
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sforward", option_prefix));
  PetscCall(PetscOptionsGetBool(NULL,NULL,option_name,&ctx->forward,&found));
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sbackward", option_prefix));
  PetscCall(PetscOptionsGetBool(NULL,NULL,option_name,&ctx->backward,&found));
  if (ctx->forward && ctx->backward) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Options -%sforward and -%sbackward cannot both be true.", option_prefix, option_prefix);
  }

  /* Step we load from */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sstep", option_prefix));
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&load_step,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
  PetscCall(LagRBFSetStep(ctx,load_step));

  /* Fields to load */
  nfields = 0;
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%snfields", option_prefix);
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&nfields,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);

  PetscCall(PetscMalloc1(nfields,&fields_name));
  nn = nfields;
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sfield_names", option_prefix));
  PetscCall(PetscOptionsGetStringArray(NULL,NULL, option_name, fields_name, &nn, &found));
  if (nfields > 0) {
    PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
    PetscCheck(nn == nfields,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,nfields,nn);
  }
  /* Load the swarm from files */
  PetscCall(SwarmLoadFromFiles(ctx, swarm_interp, restart_dir, (const char **)fields_name, nfields));
  /* Set weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(PetscFree(fields_name));

  restart_idx = 0;
  adv_scheme = -1;
  if (ctx->adv_ctx) { adv_scheme = ctx->adv_ctx->scheme; }

  ctx->forward = PETSC_TRUE;
  restart_idx  = 0;
  PetscCall(LagRBFSetTimeStep(ctx,1e-6)); // Set a small time step to start with, it will be adjusted in the time loop if necessary
  /* Move forward in time */
  PetscCall(ForwardTimeLoop(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps,restart_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Load a swarm from a previously saved state and restart the simulation from there. 
*/
static PetscErrorCode LoadRestart(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  nsteps,
  PetscInt  steps[],
  PetscReal time_steps[]
)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  PetscMPIInt      rank;
  PetscInt  load_step,nfields,nn,s,restart_idx;
  PetscBool found,exists;
  char      option_prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN],restart_dir[PETSC_MAX_PATH_LEN];
  char      **fields_name = NULL;
  const char *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Load swarm */
  PetscCall(PetscSNPrintf(option_prefix, PETSC_MAX_PATH_LEN-1, "load_"));
  /* Directory we load from */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sdirectory", option_prefix));
  PetscCall(PetscOptionsGetString(NULL,NULL,option_name,restart_dir,PETSC_MAX_PATH_LEN-1,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
  PetscCall(PetscTestDirectory(restart_dir,'r',&exists));
  PetscCheck(exists,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",restart_dir);
  
  /* Check if we load forward, backward or neither */
  ctx->forward  = PETSC_FALSE;
  ctx->backward = PETSC_FALSE;
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sforward", option_prefix));
  PetscCall(PetscOptionsGetBool(NULL,NULL,option_name,&ctx->forward,&found));
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sbackward", option_prefix));
  PetscCall(PetscOptionsGetBool(NULL,NULL,option_name,&ctx->backward,&found));
  if (ctx->forward && ctx->backward) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Options -%sforward and -%sbackward cannot both be true.", option_prefix, option_prefix);
  }

  /* Step we load from */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sstep", option_prefix));
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&load_step,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
  /* Set the new number of steps */
  for (s=0; s<nsteps; s++) {
    if (steps[s] == load_step) { 
      PetscCall(LagRBFSetStep(ctx,load_step));
      break; 
    }
  }
  if (s < nsteps - 1) { 
    /* We go from s to nsteps */
    nsteps = nsteps - s;
    restart_idx = s;
  } else {
    /* If the restart step was not found, we start from the beginning, so we set the step to 0 */
    restart_idx = 0;
  }

  /* Fields to load */
  nfields = 0;
  PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%snfields", option_prefix);
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&nfields,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);

  PetscCall(PetscMalloc1(nfields,&fields_name));
  nn = nfields;
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sfield_names", option_prefix));
  PetscCall(PetscOptionsGetStringArray(NULL,NULL, option_name, fields_name, &nn, &found));
  if (nfields > 0) {
    PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not found.",option_name);
    PetscCheck(nn == nfields,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,nfields,nn);
  }



  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunForward(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  nsteps,
  PetscInt  steps[],
  PetscReal time_steps[]
)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  PetscMPIInt      rank;
  PetscInt         s,restart_step,restart_idx;
  PetscReal        dt;
  PetscBool        exists,found;
  char             restart_dir[PETSC_MAX_PATH_LEN];
  const char       *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;

  ctx->forward  = PETSC_TRUE;
  ctx->backward = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  dt = 1.0e-7;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));

  /* Directory we load from */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-load_directory",restart_dir,PETSC_MAX_PATH_LEN-1,&found));
  PetscCall(PetscTestDirectory(restart_dir,'r',&exists));
  PetscCheck(exists,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",restart_dir);
  /* Step we restart from */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-restart_step",&restart_step,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -restart_step not found.");
  PetscCall(LagRBFSetStep(ctx,restart_step));
  /* Set the new number of steps */
  for (s=0; s<nsteps; s++) {
    if (steps[s] == restart_step) { 
      //PetscCall(LagRBFSetStep(ctx,restart_step));
      break; 
    }
  }
  if (s < nsteps - 1) { 
    /* We go from s to nsteps */
    nsteps = nsteps - s;
    restart_idx = s;
  } else {
    /* If the restart step was not found, we start from the beginning, so we set the step to 0 */
    restart_idx = 0;
  }
  //PetscPrintf(PETSC_COMM_WORLD,"nsteps = %" PetscInt_FMT ", s = %" PetscInt_FMT ", restart_idx = %" PetscInt_FMT "\n",nsteps,s,restart_idx);
  //PetscPrintf(PETSC_COMM_WORLD,"Restarting forward run from step %" PetscInt_FMT " for %" PetscInt_FMT " steps\n",steps[restart_idx],nsteps);
#if 1
  PetscCall(SwarmInterpLoadFromFile(ctx, swarm_interp, restart_dir, PETSC_TRUE));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  ctx->forward  = PETSC_TRUE;
  /* Move forward in time */
  for (s=0; s<nsteps; s++) {
    PetscInt  f,nfiles,it,step;
    PetscReal time,time_next;
    char      **filenames;

    step = steps[restart_idx + s];
    time = time_steps[restart_idx + s];
    //PetscPrintf(PETSC_COMM_WORLD,"Current idx: s = %" PetscInt_FMT ", idx = %" PetscInt_FMT ", next = %"PetscInt_FMT", nsteps = %" PetscInt_FMT "\n",s,s+restart_idx,restart_idx + s + 1,nsteps);
    if (s + 1 < nsteps) { time_next = time_steps[restart_idx + s + 1]; } 
    else                { time_next = time; }
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(StagGetFilelistFromOptions(step,&nfiles,&filenames,NULL));
    PetscPrintf(PETSC_COMM_WORLD,"FORWARD STEP[[ %" PetscInt_FMT " ]] : time = %g\n",step,time);
    PetscPrintf(PETSC_COMM_WORLD,"  Time to next step: %g\n",time_next - time);
#if 1
    for (f=0; f<nfiles; f++) {
      PetscPrintf(PETSC_COMM_WORLD,"  File[%'" PetscInt_FMT "]: %s\n",f,filenames[f]);
    }
    /* Read data only on rank 0 */
    if (rank == 0) {
      PetscCall(StagBinSwarmSetFromFiles(swarm_data,(const char**)filenames,nfiles,PETSC_FALSE));
    }
    /* Ensure that rank 0 has finished reading data */
    PetscCall(PetscBarrier((PetscObject)NULL));
    /* Send data to other ranks */
    PetscCall(DMSwarmAssignMigrateBasic(ctx,swarm_data));
    /* Duplicate ghost points */
    PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,swarm_data));
    /* Set cell IDs */
    PetscCall(DMSwarmPICSetCellId(swarm_data));
    /* Set up CellInfo data structure */
    PetscCall(CellInfoSetUp(pu,ci,swarm_data));
    /* Interpolation */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));

    /* Copy the first dim fields into "velocity" */
    PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
    it = 0;
    while (time < time_next) {
      PetscPrintf(PETSC_COMM_WORLD,"    Internal advection step [[ %" PetscInt_FMT " ]] : time = %g\n",it,time);
      /* Compute timestep */
      PetscCall(AdvectionComputeTimestep_CFL(ctx, swarm_interp, &dt));
      /* Ensure dt is positive, not sure it is useful */
      if (dt < 0.0) { dt = -dt; }
      /* Adjust timestep if necessary to not exceed the next time step */
      if (time + dt > time_next) {
        dt = time_next - time;
        PetscPrintf(PETSC_COMM_WORLD,"Timestep exceeding next step. Adjusting internal time step to %g.\n",dt);
      }
      PetscCall(LagRBFSetTimeStep(ctx,dt));
      /* Advect */
      PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
      /* Interpolate */
      PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
      PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
      /* Setup velocity for next step */
      PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
      /* Update time */
      time += dt;
      PetscCall(LagRBFSetTime(ctx,time));
      it++;
    }
    /* End of the time step */
    /* Reset partitions => clear coeffs and ksp */
    PetscCall(LagRBF_PartitionsReset(ctx));
    /* Reset cell info */
    PetscCall(CellInfoReset(pu,ci));
    /* Clear swarm data on all ranks */
    PetscCall(DMSwarmSetLocalSizes(swarm_data,0,-1));
#endif
    /* Free filenames */
    PetscCall(RestoreFilelist(nfiles,&filenames));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ForwardScalarFields(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  nsteps,
  PetscInt  steps[],
  PetscReal time_steps[]
)
{
  PetscInt restart_idx;
  PetscFunctionBeginUser;
  
  restart_idx = 0;
  PetscCall(ForwardTimeLoop(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps,restart_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Run()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscMPIInt      rank;
  PetscInt         nsteps,npoints_i;
  PetscInt         *steps;
  PetscReal        *time_steps;
  PetscBool        exists;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  /* Create the output directory */
  PetscCall(CreateDirectory(ctx->output_dir));
  
  /* Create data structures */
  ctx->mtype = STAGYY; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Create the swarm to hold data */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  
  PetscCall(GetTimeStepsFromOptions(&nsteps,&steps,&time_steps));
  PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %" PetscInt_FMT "\n",nsteps);

  /* We run forward */
  //PetscCall(RunForward(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));
  if (ctx->advection) {
    PetscCall(LoadInit(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));
  } else {
    PetscCall(ForwardScalarFields(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));
  }

  /* Clean up */
  PetscCall(RestoreTimeSteps(&steps,&time_steps));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,NULL));

  //PetscCall(Init());
  PetscCall(Run());

  PetscCall(PetscFinalize());
  
  return 0;
}