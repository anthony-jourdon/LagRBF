#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "swarm_interp.h"
#include "interpolation.h"

PetscErrorCode AdvectionComputeAdaptiveTimeStep()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS); 
}

PetscErrorCode AdvectionExplicit_DOPRI(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx adv_ctx;
  PetscInt  i,j,np,d,dim,npoints;
  PetscReal dt;
  PetscReal *coords,*velocity,*x0,*v_i;
  const char *velocity_name,**coorFieldName;
  
  PetscReal DP_coeffs[6][6] = {
    {       1.0/5.0,               0,              0,            0,               0,         0},
    {      3.0/40.0,        9.0/40.0,              0,            0,               0,         0},
    {     44.0/45.0,      -56.0/15.0,       32.0/9.0,            0,               0,         0},
    {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,               0,         0},
    { 9017.0/3168.0,     -355.0/33.0, 46732.0/5248.0,   49.0/176.0, -5103.0/18656.0,         0},
    {    35.0/384.0,               0,   500.0/1113.0,  125.0/192.0,  -2187.0/6784.0, 11.0/84.0}
  };
  PetscReal DP_weights[7] = {5179.0/57600.0, 0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetVelocityName(adv_ctx, &velocity_name));
  PetscCall(AdvectionGetTimeStep(adv_ctx,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));

  /* Store x0 and v0 = v(x0) */
  PetscCall(AdvectionSetInitialValues(adv_ctx, swi));
  /* Loop over stages of integration */
  for (i=0; i<6; i++) {
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
    PetscCall(DMSwarmGetField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
    /* Reset the velocity for the new step */
    PetscArrayzero(velocity,npoints*dim);
    for (np=0; np<npoints; np++) {
      /* compute the step velocity */
      for (j=0; j<i; j++) {
        for (d=0; d<dim; d++) {
          velocity[d*npoints + np] += DP_coeffs[i][j] * v_i[j*npoints*dim + d*npoints + np];
        }
      }
      /* Update coords */
      for (d=0; d<dim; d++) {
        coords[dim*np + d] = x0[dim*np + d] + dt * velocity[d*npoints + np];
      }
    }
    /* Restore fields */
    PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
    PetscCall(DMSwarmRestoreField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
    PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    /* Update points */
    PetscCall(DMSwarmMigrate(swi,PETSC_FALSE)); // update cell ids and MPI ranks
    PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation
    /* Interpolate */
    PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
    /* Update the velocity for the next step */
    PetscCall(DMSwarmGetField(swi, velocity_name, NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmGetField(swi, "v_adv",       NULL, NULL, (void **)&v_i));
    PetscCall(PetscArraycpy(&v_i[npoints*dim*(i+1)],velocity,npoints*dim));
    PetscCall(DMSwarmRestoreField(swi, velocity_name, NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi, "v_adv",       NULL, NULL, (void **)&v_i));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}