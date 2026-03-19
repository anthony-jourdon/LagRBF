static const char help[] = "Test to interpolate and advect particles from StagYY output.\n\n";

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
#include "readers/reader_utils.h"
#include "readers/stagYY/stag_utils.h"
#include "readers/stagYY/binary_loader.h"
#include "interpolation.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "cell_info.h"

static PetscErrorCode BackwardTimeSteps(
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
  PetscInt         s;
  PetscReal        dt;
  PetscBool        is_scalar,load,exists,found;
  PetscMPIInt      rank;
  const char       *output_fields[] = {"fields"};
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  is_scalar = PETSC_FALSE; // for now we only use velocity-pressure fields

  ctx->backward = PETSC_TRUE;
  ctx->forward  = PETSC_FALSE;

  load = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-load",&load,NULL));
  if (load) { /* We load the Swarm Interp from a checkpointed directory */
    PetscInt restart_step;
    char restart_dir[PETSC_MAX_PATH_LEN];
    /* Directory we load from */
    PetscCall(PetscOptionsGetString(NULL,NULL,"-load_directory",restart_dir,PETSC_MAX_PATH_LEN-1,&found));
    PetscCall(PetscTestDirectory(restart_dir,'r',&exists));
    PetscCheck(exists,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",restart_dir);
    /* Step we restart from */
    PetscCall(PetscOptionsGetInt(NULL,NULL,"-restart_step",&restart_step,&found));
    PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -restart_step not found.");
    /* Set the new number of steps */
    for (s=0; s<nsteps; s++) {
      if (steps[s] == restart_step) { 
        PetscCall(LagRBFSetStep(ctx,restart_step-1));
        break; 
      }
    }
    /* Assume that if it was not found, we start from the beginning */
    if (s < nsteps) { 
      /* We go from s-1 down to 0, because we do not remake the step we load from */
      nsteps = s;
    }
    PetscCall(SwarmInterpLoadFromFile(ctx, swarm_interp, restart_dir, PETSC_FALSE));
  } else { /* We create new points */
    PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,INSERT_VALUES));
  }
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));

  for (s=nsteps-1; s>=0; s--) {
    PetscInt  f,nfiles,it;
    PetscInt  step = steps[s];
    PetscReal time,time_next;
    char      **filenames;
#if 0
    if (s == nsteps-1) {
      /* Create points */
      PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,INSERT_VALUES));
      PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
    } else {
      /* Insert new points */
      PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,ADD_VALUES));
      PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
    }
#endif
    if (s < nsteps-1) {
      /* Insert new points */
      PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,ADD_VALUES));
    }

    PetscCall(LagRBFSetStep(ctx,step));
    time = time_steps[s];
    if (s > 0) { time_next = time_steps[s-1]; } 
    else       { time_next = time; }
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(StagGetFilelistFromOptions(step,&nfiles,&filenames));
    PetscPrintf(PETSC_COMM_WORLD,"BACKWARD STEP[[ %" PetscInt_FMT " ]] : time = %g\n",s,time);
    PetscPrintf(PETSC_COMM_WORLD,"  Time to next step: %g\n",time_next - time);
    for (f=0; f<nfiles; f++) {
      PetscPrintf(PETSC_COMM_WORLD,"  File[%'" PetscInt_FMT "]: %s\n",f,filenames[f]);
    }
    /* Read data only on rank 0 */
    if (rank == 0) {
      PetscCall(StagBinSwarmSetFromFiles(swarm_data,(const char**)filenames,nfiles,is_scalar));
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
    //PetscCall(SwarmOutput(ctx,swarm_data,2,output_fields,PETSC_FALSE));
    /* Interpolation */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Copy the first dim fields into "velocity" */
    PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));

    it = 0;
    while (time > time_next) {
      PetscPrintf(PETSC_COMM_WORLD,"    Internal advection step [[ %" PetscInt_FMT " ]] : time = %g\n",it,time);
      /* Compute timestep */
      PetscCall(AdvectionComputeTimestep_CFL(ctx, swarm_interp, &dt));
      if (dt > 0) { dt = -dt; } // make sure dt is negative for backward advection
      PetscCall(LagRBFSetTimeStep(ctx,dt)); 
      /* Advect */
      PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
      /* Interpolate */
      PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
      PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
      /* Setup velocity for next step */
      PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
      /* Update time */
      PetscCall(LagRBFGetTimeStep(ctx,&dt));
      if (time + dt < time_next) {
        dt = time_next - time;
        PetscCall(LagRBFSetTimeStep(ctx,dt));
      }
      time += dt;
      PetscCall(LagRBFSetTime(ctx,time));
      it++;
    }
    /* End of the time step */
    PetscCall(SwarmOutput(ctx,swarm_interp,1,output_fields,PETSC_FALSE));
    /* Reset partitions => clear coeffs and ksp */
    PetscCall(LagRBF_PartitionsReset(ctx));
    /* Reset cell info */
    PetscCall(CellInfoReset(pu,ci));
    /* Clear swarm data on all ranks */
    PetscCall(DMSwarmSetLocalSizes(swarm_data,0,-1));
    /* Free filenames */
    PetscCall(StagRestoreFilelist(nfiles,&filenames));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ForwardTimeSteps(
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
  PetscInt         s;
  PetscReal        dt;
  PetscBool        is_scalar;
  PetscMPIInt      rank;
  const char       *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  is_scalar = PETSC_FALSE; // for now we only use velocity-pressure fields
  for (s=0; s<nsteps; s++) {
    PetscInt  f,nfiles,it;
    PetscInt  step = steps[s];
    PetscReal time,time_next;
    char      **filenames;
    
    PetscCall(LagRBFSetStep(ctx,step));
    time = time_steps[s];
    if (s < nsteps-1) { time_next = time_steps[s+1]; } 
    else              { time_next = time; }
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(StagGetFilelistFromOptions(step,&nfiles,&filenames));
    PetscPrintf(PETSC_COMM_WORLD,"FORWARD STEP[[ %" PetscInt_FMT " ]] : time = %g\n",s,time);
    PetscPrintf(PETSC_COMM_WORLD,"  Time to next step: %g\n",time_next - time);
    for (f=0; f<nfiles; f++) {
      PetscPrintf(PETSC_COMM_WORLD,"  File[%'" PetscInt_FMT "]: %s\n",f,filenames[f]);
    }

    /* Read data only on rank 0 */
    if (rank == 0) {
      PetscCall(StagBinSwarmSetFromFiles(swarm_data,(const char**)filenames,nfiles,is_scalar));
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
    //PetscCall(SwarmOutput(ctx,swarm_data,2,output_fields,PETSC_FALSE));
    /* Interpolation */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Copy the first dim fields into "velocity" */
    PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
    
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));
    
    /* Advection => Internal time loop */
    it = 0;
    while (time < time_next) {
      PetscPrintf(PETSC_COMM_WORLD,"    Internal advection step [[ %" PetscInt_FMT " ]] : time = %g\n",it,time);
      /* Compute timestep */
      PetscCall(AdvectionComputeTimestep_CFL(ctx, swarm_interp, &dt));
      if (dt < 0) { dt = -dt; } // make sure dt is positive for forward advection
      PetscCall(LagRBFSetTimeStep(ctx,dt)); 
      /* Advect */
      PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
      /* Interpolate */
      PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
      PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
      /* Setup velocity for next step */
      PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
      /* Update time */
      PetscCall(LagRBFGetTimeStep(ctx,&dt));
      if (time + dt > time_next) {
        dt = time_next - time;
        PetscCall(LagRBFSetTimeStep(ctx,dt));
      }
      time += dt;
      PetscCall(LagRBFSetTime(ctx,time));
      it++;
    }
    /* If last step output after advection */
    if (s == nsteps-1) { 
      step += 1;
      PetscCall(LagRBFSetStep(ctx,step));
      PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));
    }
    /* End of the time step */
    /* Reset partitions => clear coeffs and ksp */
    PetscCall(LagRBF_PartitionsReset(ctx));
    /* Reset cell info */
    PetscCall(CellInfoReset(pu,ci));
    /* Clear swarm on all ranks */
    PetscCall(DMSwarmSetLocalSizes(swarm_data,0,-1));
    /* Free filenames */
    PetscCall(StagRestoreFilelist(nfiles,&filenames));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Run_Backward_Forward()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscInt         nsteps,npoints_i;
  PetscInt         *steps;
  PetscReal        dt;
  PetscReal        *time_steps;
  PetscBool        exists;
  PetscMPIInt      rank;
  const char       *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  
  /* Create the output directory */
  PetscCall(PetscTestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->output_dir); }

  /* Create data structures */
  ctx->mtype = GEODYNAMICS; // Set the model type
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
  
  PetscCall(StagGetTimeStepsFromOptions(&nsteps,&steps,&time_steps));
  PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %" PetscInt_FMT "\n",nsteps);

  dt = 1.0e-7;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));

  /* First we run backward */
  PetscCall(BackwardTimeSteps(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));

  /* Now that we have all our points we assign them a unique index */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  /* Output the last backward step with the unique pid as the first forward step */
  ctx->forward  = PETSC_TRUE;
  ctx->backward = PETSC_FALSE;
  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));

  /* Move forward in time */
  PetscCall(ForwardTimeSteps(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));

  /* Clean up */
  PetscCall(StagRestoreTimeSteps(&steps,&time_steps));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunBackward()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscInt         nsteps;
  PetscInt         *steps;
  PetscReal        *time_steps;
  PetscReal        dt;
  PetscBool        exists;
  PetscFunctionBeginUser;

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  
  /* Create the output directory */
  PetscCall(PetscTestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->output_dir); }

  /* Create data structures */
  ctx->mtype = GEODYNAMICS; // Set the model type
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
  
  PetscCall(StagGetTimeStepsFromOptions(&nsteps,&steps,&time_steps));
  PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %" PetscInt_FMT "\n",nsteps);

  dt = 1.0e-7;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));

  /* First we run backward */
  PetscCall(BackwardTimeSteps(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));

  /* Clean up */
  PetscCall(StagRestoreTimeSteps(&steps,&time_steps));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,help));

  //PetscCall(Run_Backward_Forward());
  PetscCall(RunBackward());

  PetscCall(PetscFinalize());
  
  return 0;
}
