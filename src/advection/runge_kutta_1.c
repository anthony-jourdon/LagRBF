#include <petsc.h>
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "advection/runge_kutta_1.h"
#include "lagrangian_rbf.h"
#include "swarm/swarm_interp.h"

PetscLogEvent LAGRBF_AdvectionExplicit_RK1;

PetscErrorCode AdvectionExplicit_RK1(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv;
  PetscInt      dim,npoints;
  PetscReal     dt;
  PetscReal     *x0,*coords,*velocity;
  const char    *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_AdvectionExplicit_RK1,0,0,0,0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME));

  /* Get advection context and time step */
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv));
  PetscCall(AdvectionGetTimeStep(adv,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(adv, &velocity_name));
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,   NULL, NULL,(void **)&velocity));
  /* Store initial values */
  PetscCall(PetscMalloc1(npoints*dim,&x0));
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* Advect */
  PetscCall(AdvectionExplicit(coords,x0,velocity,dt,npoints,dim));
  PetscCall(PetscFree(x0));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

  PetscCall(PetscLogEventEnd(LAGRBF_AdvectionExplicit_RK1,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Advection_RK1Create(AdvectionCtx adv_ctx)
{
  PetscFunctionBeginUser;
  adv_ctx->advection = AdvectionExplicit_RK1;
  adv_ctx->destroy   = NULL;
  adv_ctx->order     = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}