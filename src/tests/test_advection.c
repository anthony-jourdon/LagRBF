static const char help[] = "Testing advection schemes.\n\n";

#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "partitions/partition_solve.h"
#include "cell_info.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "interpolation.h"
#include "advection/advection_ctx.h"
#include "swarm/swarm_output.h"

static PetscErrorCode RigidRotationVelocity_3D(PetscReal coords[], PetscReal velocity[], void *ctx)
{
  PetscReal *angular_u = (PetscReal*)ctx;
  PetscReal r,theta,phi;
  PetscFunctionBeginUser;

  r = PetscSqrtReal(coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2]);
  theta = PetscAtan2Real(PetscSqrtReal(coords[0]*coords[0] + coords[1]*coords[1]), coords[2]);
  phi   = PetscAtan2Real(coords[1], coords[0]);
  
  velocity[0] = angular_u[1] * r * PetscCosReal(theta) - angular_u[2] * r * PetscSinReal(theta) * PetscSinReal(phi);
  velocity[1] = angular_u[2] * r * PetscSinReal(theta) * PetscCosReal(phi) - angular_u[0] * r * PetscCosReal(theta);
  velocity[2] = angular_u[0] * r * PetscSinReal(theta) * PetscSinReal(phi) - angular_u[1] * r * PetscSinReal(theta) * PetscCosReal(phi);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RegisterRigidRotationVelocity(AdvectionCtx ctx, PetscReal *angular_u)
{
  PetscInt  nn,d;
  PetscReal angular_vel;
  PetscReal rotation_vec[] = {0.0, 0.0, 1.0}; // Default rotation around z-axis
  PetscBool found;
  PetscFunctionBeginUser;

  angular_vel = 0.5;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-angular_velocity",&angular_vel,NULL));
  nn = 3;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-rotation_vector",rotation_vec,&nn,&found));
  if (found) { PetscCheck(nn == 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -rotation_vector must have 3 values, found: %" PetscInt_FMT "",nn); }

  for (d=0; d<3; d++) { angular_u[d] = angular_vel * rotation_vec[d]; }
  PetscCall(AdvectionSetVelocityFunction(ctx,RigidRotationVelocity_3D,(void *)angular_u));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_RigidRotation3D()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_interp;
  PetscInt         step,npoints_i,max_steps;
  PetscReal        dt, time_final, time;
  PetscReal        angular_u[3];
  PetscFunctionBeginUser;

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

  /* Advection context */
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(RegisterRigidRotationVelocity(adv_ctx, angular_u));

  /* Create the swarm */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  PetscCall(DMSwarmSetPointsCoordinates_FromOptions(swarm_interp,INSERT_VALUES));
  /* Set weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  
  /* Timestep */
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));

  /* Time limit */
  time_final = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_final",&time_final,NULL));
  time = 0.0;
  step = 0;
  PetscCall(LagRBFSetStep(ctx,step));
  PetscCall(LagRBFSetTime(ctx,time));
  max_steps = (PetscInt)(time_final/dt);

  /* Initialize velocity */
  PetscCall(AdvectionInitializeTimestepVelocity(ctx, swarm_interp));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));

  for (step=0; step<max_steps; step++) {
    PetscPrintf(PETSC_COMM_WORLD,"    Advection step [[ %" PetscInt_FMT " ]] : time = %1.4e, dt = %1.4e\n",step,time,dt);
    /* Advect */
    PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,NULL));
    /* Initialize velocity for next timestep */
    PetscCall(AdvectionInitializeTimestepVelocity(ctx, swarm_interp));
    /* Update time */
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(LagRBFSetStep(ctx,step));
  }
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));
  /* Free memory */
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,help));

  PetscCall(Test_RigidRotation3D());

  PetscCall(PetscFinalize());
  
  return 0;
}