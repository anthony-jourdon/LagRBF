#include <petsc.h>
#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "advection/runge_kutta_4.h"
#include "swarm_interp.h"
#include "interpolation.h"

PetscLogEvent LAGRBF_AdvectionExplicit_RK4;
/*
  1- Advect points with the velocity from the previous stage: 
    x_s = x0 + dt * v(x_{s-1})
  2- Update points in the DMSwarm for cell id and MPI rank
  3- Interpolate the velocity at the new position x_s to get v_s = v(x_s)
    v_s = v(x_s)
  4- Store the velocity v_s in the field "v_adv" for the next stage
*/
static PetscErrorCode AdvectionExplicit_SingleStage(LagRBFCtx ctx, DM swi, DM swd, PetscInt stage, PetscReal dt)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv;
  PetscInt      dim,npoints;
  PetscReal     *coords,*velocity,*x0,*v_i;
  const char    *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(adv, &velocity_name));
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  /* Copy velocity from previous stage */
  PetscCall(PetscArraycpy(velocity,&v_i[(stage-1)*npoints*dim],npoints*dim));
  /* Advect points */
  PetscCall(AdvectionExplicit(coords,x0,velocity,dt,npoints,dim));
  //PetscCall(AdvectionExplicit_Vec(coords,x0,velocity,dt,npoints,dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation
  /* Interpolate v_s = v(x_s) with s = stage */
  PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
  /* Update the velocity for the next step */
  PetscCall(DMSwarmGetField(swi, velocity_name, NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "v_adv",       NULL, NULL, (void **)&v_i));
  PetscCall(PetscArraycpy(&v_i[npoints*dim*stage],velocity,npoints*dim));
  PetscCall(DMSwarmRestoreField(swi, velocity_name, NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "v_adv",       NULL, NULL, (void **)&v_i));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionExplicit_RK4(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv;
  PetscInt      s,dim,npoints,np,d;
  PetscReal     dt;
  PetscReal     RK4_coeffs[]  = {0.5, 0.5, 1.0};
  PetscReal     RK4_weights[] = {1.0, 2.0, 2.0, 1.0};
  PetscReal     *coords,*velocity,*x0,*v_i;
  const char    *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_AdvectionExplicit_RK4,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv));
  PetscCall(AdvectionGetTimeStep(adv,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(adv, &velocity_name));
  /* Step 1: Store initial position (x0) and velocity v0 = v(x0) */
  PetscCall(AdvectionSetInitialValues(adv,swi));
  /* Step 2: Interpolate v1, v2 and v3 */
  for (s=0; s<3; s++) {
    PetscInt stage = s + 1;
    PetscCall(AdvectionExplicit_SingleStage(ctx,swi,swd,stage,RK4_coeffs[s]*dt));
  }
  /* Step 3: Combine stages and advect */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  for (np=0; np<npoints; np++) {
    PetscReal sum[] = {0.0, 0.0, 0.0};
    for (d=0; d<dim; d++) {
      for (s=0; s<4; s++) {
        sum[d] += RK4_weights[s] * v_i[s*npoints*dim + d*npoints + np];
      }
      coords[dim*np + d] = x0[dim*np + d] + dt/6.0 * sum[d];
    }
  }
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

  PetscCall(PetscLogEventEnd(LAGRBF_AdvectionExplicit_RK4,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Advection_RK4Create(AdvectionCtx adv_ctx)
{
  PetscFunctionBeginUser;
  adv_ctx->advection = AdvectionExplicit_RK4;
  adv_ctx->destroy   = NULL;
  adv_ctx->order     = 4;
  PetscFunctionReturn(PETSC_SUCCESS);
}