#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "cell_info.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_output.h"
#include "swarm/swarm_load.h"
#include "interpolation.h"

PetscErrorCode InternalForwardAdvectionLoop(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscReal time,
  PetscReal time_next
)
{
  AdvectionScheme  adv_scheme;
  PartitionOfUnity pu;
  CellInfo        *ci;
  PetscInt         it;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  PetscCall(LagRBFGetAdvectionScheme(ctx,&adv_scheme));

  /* Copy the first dim fields into "velocity" */
  PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
  it = 0;
  while (time < time_next) {
    PetscReal dt;

    PetscPrintf(PETSC_COMM_WORLD,"    Internal advection step [[ %" PetscInt_FMT " ]] : time = %1.4e\n",it,time);
    /* Compute timestep */
    if (adv_scheme == DOPRI) {
      PetscReal dt_cfl, dt_dopri;
      PetscCall(LagRBFGetTimeStep(ctx,&dt_dopri));
      PetscCall(AdvectionComputeTimestep_CFL(ctx, swarm_interp, &dt_cfl));
      dt = PetscMin(dt_dopri, dt_cfl);
      PetscPrintf(PETSC_COMM_WORLD,"  DOPRI time step: %1.4e, CFL time step: %1.4e, selected time step: %1.4e\n",dt_dopri, dt_cfl, dt);
    } else {
      PetscCall(AdvectionComputeTimestep_CFL(ctx, swarm_interp, &dt));
    }
    /* Adjust timestep if necessary to not exceed the next time step */
    if (time + dt > time_next) {
      dt = time_next - time;
      PetscPrintf(PETSC_COMM_WORLD,"Timestep exceeding next step. Adjusting internal time step to %1.4e.\n",dt);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ForwardTimeLoop(
  LagRBFCtx ctx,
  DM        swarm_data,
  DM        swarm_interp,
  PetscInt  nsteps,
  PetscInt  steps[],
  PetscReal time_steps[],
  PetscInt  restart_idx
)
{
  PetscInt s;
  PetscFunctionBeginUser;

  ctx->forward  = PETSC_TRUE;
  ctx->backward = PETSC_FALSE;
  /* Move forward in time */
  for (s=0; s<nsteps; s++) {
    PetscInt step;
    PetscReal time, time_next;

    step = steps[restart_idx + s];
    time = time_steps[restart_idx + s];

    if (s + 1 < nsteps) { time_next = time_steps[restart_idx + s + 1]; } 
    else                { time_next = time; }
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFSetTime(ctx,time));
    PetscPrintf(PETSC_COMM_WORLD,"FORWARD STEP[[ %" PetscInt_FMT " ]] : time = %g\n",step,time);
    PetscPrintf(PETSC_COMM_WORLD,"  Time to next step: %g\n",time_next - time);

    /* If there is no advection we load the swarm on which we interpolate */
    if (!ctx->advection) {
      PetscBool found;
      char      directory[PETSC_MAX_PATH_LEN];

      /* Clear swarm interp on all ranks */
      PetscCall(DMSwarmSetLocalSizes(swarm_interp,0,-1));
      PetscCall(PetscSNPrintf(directory, PETSC_MAX_PATH_LEN-1, "%s/step%" PetscInt_FMT, ctx->load_dir, step));
      PetscCall(PetscTestDirectory(directory,'r',&found));
      PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Load directory %s not found.",directory);
      PetscCall(SwarmLoadFromFiles(ctx, swarm_interp, (const char *)directory, (const char **)ctx->load_fields, ctx->nload));
      /* Set weights for interpolation */
      PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
    }
    /* Load data on swarm */
    PetscCall(SwarmDataLoad(ctx, swarm_data));
    /* Interpolation */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));
    /* Advection */
    if (ctx->advection) {
      PetscCall(InternalForwardAdvectionLoop(ctx,swarm_data,swarm_interp,time,time_next));
    } else {
      PartitionOfUnity pu;
      CellInfo        *ci;
      /* End of the time step */
      PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
      PetscCall(LagRBFGetCellInfo(ctx,&ci));
      /* Reset partitions => clear coeffs and ksp */
      PetscCall(LagRBF_PartitionsReset(ctx));
      /* Reset cell info */
      PetscCall(CellInfoReset(pu,ci));
      /* Clear swarm data on all ranks */
      PetscCall(DMSwarmSetLocalSizes(swarm_data,0,-1));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}