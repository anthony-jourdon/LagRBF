static const char help[] = "Testing functionalities in parallel.\n\n";

#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "io_utils.h"
#include "swarm/swarm_interp.h"

static PetscErrorCode AnalyticalRotationVelocity(PetscReal coor[], PetscReal velocity[])
{
  PetscReal r,theta;
  PetscReal radial_u  = 0.0;
  PetscReal angular_u = 0.5;
  PetscFunctionBeginUser;
  r     = PetscSqrtReal(coor[0]*coor[0] + coor[1]*coor[1]);
  theta = PetscAtan2Real(coor[1], coor[0]);
  velocity[0] = radial_u * PetscCosReal(theta) - angular_u * r * PetscSinReal(theta);
  velocity[1] = radial_u * PetscSinReal(theta) + angular_u * r * PetscCosReal(theta);  
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode SwarmSetVelocity(DM swarm)
{
  DMSwarmCellDM celldm;
  PetscInt      npoints,dim,p;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  for (p=0; p<npoints; p++) {
    PetscCall(AnalyticalRotationVelocity(&coords[dim*p], &velocity[dim*p]));
  }
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Advect_RK1(DM swarm, PetscReal dt)
{
  DMSwarmCellDM celldm;
  PetscInt      p,dim,npoints,d;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  for (p=0; p<npoints; p++) {
    for (d=0; d<dim; d++) {
      coords[dim*p + d] += dt * velocity[dim*p + d];
    }
  }
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  DM               dap,swarm;
  PetscInt         dim,ppcell,step;
  PetscReal        dt,time,time_end;
  PetscBool        exists;
  char             dirname[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(DMGetDimension(ctx->pu->dap,&dim));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));

  /* Create Swarm */
#if 0
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dap), &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, dap));
  PetscCall(DMSetFromOptions(swarm));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"velocity",dim,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));
#endif
  PetscCall(SwarmInterpCreate(ctx,&swarm));
  ppcell = 8;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL));
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm, DMSWARMPIC_LAYOUT_REGULAR, ppcell));
  /* Set velocity */
  PetscCall(SwarmSetVelocity(swarm));
  /* Output */
  PetscCall(PetscSNPrintf(dirname,PETSC_MAX_PATH_LEN-1,"%s",ctx->output_dir));
  PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/swarm_advection_0.xmf",dirname));
  //PetscCall(DMSwarmViewXDMF(swarm,fname));
  /* Advect */
  dt = 1e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  step = 1;
  time = dt;
  time_end = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_end",&time_end,NULL));
  while (time < time_end) {
    PetscPrintf(PETSC_COMM_WORLD,"Step: %" PetscInt_FMT " Time: %g\n",step,time);
    PetscCall(Advect_RK1(swarm, dt));
    time += dt;
    /* Update velocity */
    PetscCall(SwarmSetVelocity(swarm));
    /* Output */
    PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/swarm_advection_%" PetscInt_FMT ".xmf",dirname,step));
    //PetscCall(DMSwarmViewXDMF(swarm,fname));
    step++;
  }
  PetscCall(DMDestroy(&swarm));
  PetscCall(LagRBFCtxDestroy(&ctx));
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