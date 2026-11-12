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
  file:    run_forward.c

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

static PetscErrorCode CreateNewSwarmInterp()
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Creating a new swarm to interpolate on is not implemented yet. Only loading from files is currently supported.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestartFromSwarmInterp(
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
  PetscInt         load_step,f;
  PetscBool        found;
  char             option_prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN],restart_dir[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  PetscCall(PetscSNPrintf(option_prefix, PETSC_MAX_PATH_LEN-1, "load_"));

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

  PetscCall(PetscSNPrintf(restart_dir, PETSC_MAX_PATH_LEN-1, "%s/step%" PetscInt_FMT, ctx->load_dir, load_step));
  PetscCall(PetscTestDirectory(restart_dir,'r',&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",restart_dir);
  /* Load the swarm from files */
  PetscCall(SwarmLoadFromFiles(ctx, swarm_interp, (const char *)restart_dir, (const char **)ctx->load_fields, ctx->nload));
  /* Set weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  /* Check if DMSwarm_pid has been loaded, otherwise assign unique indices */
  found = PETSC_FALSE;
  for (f=0; f<ctx->nload; f++) {
    PetscStrcmp(ctx->load_fields[f], DMSwarmField_pid, &found);
    if (found) { break; }
  }
  if (!found) {
    PetscInt npoints_i;
    PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
    PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  }
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
  PetscInt         nsteps,restart_idx;
  PetscInt         *steps;
  PetscReal        dt;
  PetscReal        *time_steps;
  PetscBool        found;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  /* Create the output directory */
  PetscCall(CreateDirectory(ctx->output_dir));
  
  /* Create data structures */
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
  if (ctx->advection) {
    PetscCall(RestartFromSwarmInterp(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps));
    PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
    PetscCall(LagRBFSetTimeStep(ctx,dt)); // Set a small time step to start with, it will be adjusted in the time loop if necessary
  }
  /* Get the index of the step we load from */
  restart_idx = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-load_restart_index",&restart_idx,&found));
  if (!found) { PetscPrintf(PETSC_COMM_WORLD,"Option -load_restart_index not found, use it to change the first step from which to start the run forward.\n"); }
  PetscPrintf(PETSC_COMM_WORLD,"Starting from step[%" PetscInt_FMT "] = %" PetscInt_FMT ".\n", restart_idx, steps[restart_idx]);
  
  /* Move forward in time */
  PetscCall(ForwardTimeLoop(ctx,swarm_data,swarm_interp,nsteps,steps,time_steps,restart_idx));

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

  PetscCall(Run());

  PetscCall(PetscFinalize());
  
  return 0;
}