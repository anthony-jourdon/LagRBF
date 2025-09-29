#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "advection/dormand_prince.h"
#include "swarm/swarm_interp.h"
#include "interpolation.h"

PetscLogEvent LAGRBF_AdvectionExplicit_DOPRI;

PetscErrorCode AdvectionEvaluateError_DOPRI(AdvectionCtx ctx, DM swi)
{
  DoPriCtx      *data = (DoPriCtx *)ctx->user;
  DMSwarmCellDM celldm;
  PetscInt      np,i,d,npoints,dim;
  PetscReal     dt,max_err,a_tol,r_tol;
  PetscReal     *coords,*x0;
  PetscReal     *v_i[] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL}; // array to store velocities at each stage
  char          v_adv_name[PETSC_MAX_PATH_LEN];
  const char    **coorFieldName;
  PetscReal     DP_weights[7] = {5179.0/57600.0, 0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(AdvectionGetTimeStep(ctx,&dt));

  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  for (i=0; i<7; i++) {
    PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",i));
    PetscCall(DMSwarmGetField(swi, v_adv_name, NULL, NULL, (void **)&v_i[i]));
  }
  /* Compute error between two orders of integration */
  max_err = PETSC_MIN_REAL; // Initialize max_err to a very small value
  a_tol   = data->a_tol;
  r_tol   = data->r_tol;
  for (np=0; np<npoints; np++) {
    PetscReal *v_high = v_i[6]; // last stage
    PetscReal v_low[] = {0.0, 0.0, 0.0}; // low order velocity
    PetscReal E,max_value,error,tol;
    /* Compute the low order velocity */
    for (i=0; i<7; i++) {
      for (d=0; d<dim; d++) {
        v_low[d] += DP_weights[i] * v_i[i][dim*np + d];
      }
    }
    E = 0.0;
    for (d=0; d<dim; d++) {
      max_value = PetscMax(PetscAbs(x0[dim*np + d]), PetscAbs(coords[dim*np + d]));
      /* Get tolerance */
      tol = a_tol + r_tol * max_value;
      /* Compute the error */
      error = dt * (v_high[dim*np + d] - v_low[d]);
      E += (error * error) / (tol * tol);
    }
    E = PetscSqrtReal(1.0/dim * E);
    max_err = PetscMax(max_err, E);
  }
  /* Perform reduction to find global max error */
  PetscCall(MPI_Allreduce(&max_err, &data->error, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
  //data->error = max_err; 
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  for (i=0; i<7; i++) {
    PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",i));
    PetscCall(DMSwarmRestoreField(swi, v_adv_name, NULL, NULL, (void **)&v_i[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If error is small we take the time step */ 
static PetscErrorCode AdvectionTakeStep_DOPRI(LagRBFCtx ctx, DM swi)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv_ctx;
  DoPriCtx      *data;
  PetscInt      npoints,dim;
  PetscReal     dt,dt_new,fac,fac_min,fac_max,max_err;
  const char    **coorFieldName;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  data = (DoPriCtx *)adv_ctx->user;
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(AdvectionGetTimeStep(adv_ctx,&dt));
  /* Update MPI rank and cell id */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE));
  /* Update weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swi));
  /* Compute the new time step */
  fac     = data->dt_fac;
  fac_min = data->dt_fac_min;
  fac_max = data->dt_fac_max;
  max_err = data->error;
  dt_new  = dt * PetscMin( fac_max, PetscMax( fac_min, fac * PetscPowReal( 1.0/max_err, 1.0/6.0 ) ) );
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Time step accepted with max error: %1.4e, new dt: %1.4e\n",PETSC_FUNCTION_NAME,max_err,dt_new);
  PetscCall(AdvectionSetTimeStep(adv_ctx,dt_new));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectionRejectStep_DOPRI(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv_ctx;
  DoPriCtx      *data;
  PetscInt      npoints,dim;
  PetscReal     *coords,*x0,*v0,*velocity;
  PetscReal     dt,dt_new,fac,fac_min,fac_max,max_err;
  const char    **coorFieldName,*velocity_name;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  data = (DoPriCtx *)adv_ctx->user;
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(AdvectionGetTimeStep(adv_ctx,&dt));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));

  fac     = data->dt_fac;
  fac_min = data->dt_fac_min;
  fac_max = 1.0;
  max_err = data->error;
  dt_new  = dt * PetscMin( fac_max, PetscMax( fac_min, fac * PetscPowReal( 1.0/max_err, 1.0/6.0 ) ) );
  PetscCall(AdvectionSetTimeStep(adv_ctx,dt_new));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Time step rejected with max error: %1.4e, new dt: %1.4e\n",PETSC_FUNCTION_NAME,max_err,dt_new);
  /* Reset coordinates and velocity to initial values */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "v_adv_0",        NULL, NULL, (void **)&v0));
  PetscCall(PetscArraycpy(coords,x0,npoints*dim)); // Reset coordinates to initial values
  PetscCall(PetscArraycpy(velocity,v0,npoints*dim)); // Reset velocity to initial values
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "v_adv_0",        NULL, NULL, (void **)&v0));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation
  /* Re-do a Dormand-Prince integration */
  PetscCall(AdvectionExplicit_DOPRI(ctx,swi,swd)); 
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionComputeAdaptiveTimeStep_DOPRI(LagRBFCtx ctx, DM swi, DM swd)
{
  AdvectionCtx adv_ctx;
  DoPriCtx     *data;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  data = (DoPriCtx *)adv_ctx->user;

  if (data->error <= 1.0) {
    PetscCall(AdvectionTakeStep_DOPRI(ctx,swi));
  } else {
    PetscCall(AdvectionRejectStep_DOPRI(ctx,swi,swd));
  }

  PetscFunctionReturn(PETSC_SUCCESS); 
}

PetscErrorCode AdvectionExplicit_DOPRI(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv_ctx;
  PetscInt      npoints,dim,s;
  PetscReal     dt;
  PetscReal     *coords,*velocity,*x0,*v_stage,*v_high;
  const char    *velocity_name,**coorFieldName;
  PetscReal     DP_coeffs[6][6] = {
    {       1.0/5.0,               0,              0,            0,               0,         0},
    {      3.0/40.0,        9.0/40.0,              0,            0,               0,         0},
    {     44.0/45.0,      -56.0/15.0,       32.0/9.0,            0,               0,         0},
    {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,               0,         0},
    { 9017.0/3168.0,     -355.0/33.0, 46732.0/5248.0,   49.0/176.0, -5103.0/18656.0,         0},
    {    35.0/384.0,               0,   500.0/1113.0,  125.0/192.0,  -2187.0/6784.0, 11.0/84.0}
  };
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_AdvectionExplicit_DOPRI,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  PetscCall(AdvectionGetTimeStep(adv_ctx,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm,NULL,&coorFieldName));

  /* Store x0 and v0 = v(x0) */
  PetscCall(AdvectionSetInitialValues(adv_ctx, swi));
  /* Stages 1 to 6 */
  for (s=0; s<6; s++) {
    PetscInt  stage,np,j,d;
    PetscReal *v_i[] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL}; // array to store velocities at each stage
    char      v_adv_name[PETSC_MAX_PATH_LEN];

    stage = s + 1; // stage 0 is already set by AdvectionSetInitialValues
    PetscCall(DMSwarmGetLocalSize(swi,&npoints));
    /* Access coordinates and velocity */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
    /* Access velocity of the required previous steps */
    for (j=0; j<stage; j++) {
      PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",j));
      PetscCall(DMSwarmGetField(swi, v_adv_name, NULL, NULL, (void **)&v_i[j]));
    }
    /* Reset velocity */
    PetscArrayzero(velocity,npoints*dim);
    for (np=0; np<npoints; np++){
      /* Compute the step velocity using previous stages */
      for (j=0; j<stage; j++) {
        for (d=0; d<dim; d++) {
          velocity[dim*np + d] += DP_coeffs[s][j] * v_i[j][dim*np + d];
        }
      }
      /* Update coordinates */
      for (d=0; d<dim; d++) {
        coords[dim*np + d] = x0[dim*np + d] + dt * velocity[dim*np + d];
      }
    }
    /* Restore fields */
    PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
    for (j=0; j<stage; j++) {
      PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",j));
      PetscCall(DMSwarmRestoreField(swi, v_adv_name, NULL, NULL, (void **)&v_i[j]));
    }
    /* Update points */
    PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
    PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation
    /* Interpolate */
    PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
    /* Store velocity for the current step */
    PetscCall(DMSwarmGetLocalSize(swi,&npoints));
    PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",stage));
    PetscCall(DMSwarmGetField(swi, v_adv_name, NULL, NULL, (void **)&v_stage));
    PetscCall(DMSwarmGetField(swi, velocity_name, NULL, NULL, (void **)&velocity));
    PetscCall(PetscArraycpy(v_stage,velocity,npoints*dim));
    PetscCall(DMSwarmRestoreField(swi, velocity_name, NULL, NULL, (void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi, v_adv_name, NULL, NULL, (void **)&v_stage));
  }
  /* Advect with high order velocity */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(swi, "v_adv_6",        NULL, NULL, (void **)&v_high));
  PetscCall(AdvectionExplicit(coords,x0,v_high,dt,npoints,dim));
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, "v_adv_6",        NULL, NULL, (void **)&v_high));
  /* Evaluate error */
  PetscCall(AdvectionEvaluateError_DOPRI(adv_ctx,swi));
  /* Compute adaptive time step and decide if step is taken or rejected */
  PetscCall(AdvectionComputeAdaptiveTimeStep_DOPRI(ctx,swi,swd));

  PetscCall(PetscLogEventEnd(LAGRBF_AdvectionExplicit_DOPRI,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DOPRICtxCreate(DoPriCtx **data)
{
  DoPriCtx *ctx;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&ctx));
  ctx->dt_fac     = 0.9;    // default factor for time step adaptation
  ctx->dt_fac_min = 1.0e-2; // minimum factor for time step adaptation
  ctx->dt_fac_max = 1.0e2;  // maximum factor for time step adaptation
  ctx->a_tol      = 1.0e-3; // absolute tolerance for error evaluation
  ctx->r_tol      = 1.0e-3; // relative tolerance for error evaluation
  ctx->error      = PETSC_MAX_REAL; // initialize error to a large value

  *data = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DOPRICtxDestroy(AdvectionCtx adv_ctx)
{
  DoPriCtx *data = (DoPriCtx *)adv_ctx->user;
  PetscFunctionBeginUser;
  if (data) {
    PetscCall(PetscFree(data));
    adv_ctx->user = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DOPRICtxSetFromOptions(DoPriCtx *data)
{
  char prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"dopri_"));
  PetscCall(PetscOptionsGetReal(NULL,prefix,"-dt_fac",&data->dt_fac,NULL));
  PetscCall(PetscOptionsGetReal(NULL,prefix,"-dt_fac_min",&data->dt_fac_min,NULL));
  PetscCall(PetscOptionsGetReal(NULL,prefix,"-dt_fac_max",&data->dt_fac_max,NULL));
  PetscCall(PetscOptionsGetReal(NULL,prefix,"-atol",&data->a_tol,NULL));
  PetscCall(PetscOptionsGetReal(NULL,prefix,"-rtol",&data->r_tol,NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Advection_DOPRICreate(AdvectionCtx adv_ctx)
{
  DoPriCtx *data = NULL;
  PetscFunctionBeginUser;
  PetscCall(DOPRICtxCreate(&data));
  PetscCall(DOPRICtxSetFromOptions(data));
  adv_ctx->advection = AdvectionExplicit_DOPRI;
  adv_ctx->destroy   = DOPRICtxDestroy;
  adv_ctx->order     = 7;
  adv_ctx->user      = (void*)data;
  PetscFunctionReturn(PETSC_SUCCESS);
}