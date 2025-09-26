#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm_data.h"
#include "interpolation.h"
#include "swarm/swarm_output.h"

static PetscReal triple_gaussians(PetscReal x, PetscReal y)
{
  /*
    0.81*exp(-0.9*(x + 1.0)**2 - 0.9*(z + 1.0)**2)
    1.0*exp(-4.0*x**2 - 4.0*(z - 1.5)**2)
    0.5625*exp(-0.75*z**2 - 0.75*(x - 1.5)**2)
  */
  PetscReal value;
  PetscFunctionBeginUser;
  
  value = 0.81*PetscExpReal(-0.9*((x + 1.0)*(x + 1.0) + (y + 1.0)*(y + 1.0)));
  value += 1.0*PetscExpReal(-4.0*(x*x + (y - 1.5)*(y - 1.5)));
  value += 0.5625*PetscExpReal(-0.75*((x - 1.5)*(x - 1.5) + y*y));
  PetscFunctionReturn(value);
}

static PetscReal circle_gaussian(PetscReal x, PetscReal y)
{
  PetscReal value;
  PetscFunctionBeginUser;
  value = 0.5*PetscExpReal(-3.0*pow(PetscSqrtReal(x*x + y*y) - 1.0, 2));
  PetscFunctionReturn(value);
}

static PetscReal discontinuous_cross_planes(PetscReal x, PetscReal y)
{
  PetscReal sqrt_2_o_2 = 0.70710678118654757; // sqrt(2)/2
  PetscFunctionBeginUser;
  if (sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
    if (-sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
      PetscFunctionReturn(1.0); // first quadrant
    } else {
      PetscFunctionReturn(2.0); // second quadrant
    }
  } else {
    if (-sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
      PetscFunctionReturn(3.0); // third quadrant
    } else {
      PetscFunctionReturn(4.0); // fourth quadrant
    }
  }
}

static PetscErrorCode velocity_function(PetscReal coor[], PetscReal velocity[])
{
  PetscReal r,theta;
  PetscReal radial_u  = 0.0; // -0.2;
  PetscReal angular_u = 0.5;
  PetscFunctionBeginUser;
  r     = PetscSqrtReal(coor[0]*coor[0] + coor[1]*coor[1]);
  theta = PetscAtan2Real(coor[1], coor[0]);
  velocity[0] = radial_u * PetscCosReal(theta) - angular_u * r * PetscSinReal(theta);
  velocity[1] = radial_u * PetscSinReal(theta) + angular_u * r * PetscCosReal(theta);  
  PetscFunctionReturn(PETSC_SUCCESS); // return the magnitude of the velocity
}

static PetscErrorCode CreateDataPoints(LagRBFCtx ctx, DM *sw)
{
  DM         swarm;
  PetscInt   dim,nn;
  PetscInt   *m;
  PetscReal  O[]={0.0, 0.0, 0.0};
  PetscReal  L[]={1.0, 1.0, 1.0};
  PetscBool  found;
  char       option_name[PETSC_MAX_PATH_LEN];
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(SwarmDataCreate(ctx,&swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(PetscMalloc1(dim,&m));
  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",prefix));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,m,&nn,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%s not found. User must provide the number of data points in each dimension.",option_name);
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,nn);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-O",O,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sO must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-L",L,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sL must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscCall(PetscFree(m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmDataSetValues(DM sw)
{
  DMSwarmCellDM celldm;
  PetscInt      p,bs,dim,npoints;
  PetscReal     *coords,*fields;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMGetDimension(sw,&dim));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(sw, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,"fields",&bs,NULL,(void **)&fields));
  for (p=0; p<npoints; p++) {
    /* We will assume that the velocity is the ndim first field dof */
    PetscCall(velocity_function(&coords[dim*p],&fields[bs*p]));
    fields[bs*p + dim  ] = triple_gaussians(coords[dim*p], coords[dim*p+1]); // first field
    fields[bs*p + dim+1] = circle_gaussian(coords[dim*p], coords[dim*p+1]); // second field
    fields[bs*p + dim+2] = discontinuous_cross_planes(coords[dim*p], coords[dim*p+1]); // third field
  }
  PetscCall(DMSwarmRestoreField(sw,"fields",&bs,NULL,(void **)&fields));
  PetscCall(DMSwarmRestoreField(sw,coorFieldName[0],NULL,NULL,(void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointAdvection_Euler_Pointwise(DM sw, PetscReal dt, const char *velocity_name)
{
  DMSwarmCellDM  celldm;
  PetscInt       p,dim,bs,npoints;
  PetscReal      *coords,*velocity;
  PetscLogDouble time[2];
  const char     **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(DMGetDimension(sw,&dim));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(sw, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,velocity_name,&bs,NULL,(void **)&velocity));
  for (p=0; p<npoints; p++) {
    PetscInt d;
    for (d=0; d<dim; d++) {
      coords[dim*p + d] = coords[dim*p + d] + velocity[bs*p + d] * dt;
    }
  }
  PetscCall(DMSwarmRestoreField(sw,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(sw,coorFieldName[0],NULL,NULL,(void **)&coords));
  /* Update MPI rank and cell id */
  PetscCall(DMSwarmMigrate(sw,PETSC_TRUE));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  1. Compute x1 = x0 + 0.5*dt * v(x0)
  2. Interpolate v at x1 to get v(x1)
  3. Compute x2 = x0 + 0.5*dt * v(x1)
  4. Interpolate v at x2 to get v(x2)
  5. Compute x3 = x0 + dt * v(x2)
  6. Interpolate v at x3 to get v(x3)
  7. Compute x_final = x0 + (dt/6) * (v(x0) + 2*v(x1) + 2*v(x2) + v(x3))
  */
static PetscErrorCode AdvectionExplicit_SingleVelocityStage(
  LagRBFCtx ctx,
  DM swi, 
  DM swd,
  PetscReal dt, 
  PetscInt v_idx, 
  PetscInt stage_idx,
  const char *velocity_name
)
{
  DMSwarmCellDM  celldm;
  PetscInt       np,d,bs,x0_bs,v_bs,coor_bs,dim,npoints;
  PetscReal      *coords,*x0,*velocity,*v_i;
  const char     **coorFieldName;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));

  PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
  PetscCheck(x0_bs == coor_bs,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The initial coordinates field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",coor_bs,x0_bs);
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  /* Advect */
  for (np=0; np<npoints; np++) {
    for (d=0; d<dim; d++) {
      velocity[d*npoints + np] = v_i[v_idx*npoints*dim + d*npoints + np]; // get the velocity for the current stage
      coords[dim*np + d] = x0[dim*np + d] + dt * velocity[d*npoints + np];
    }
  }
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));

  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_FALSE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

  /* Interpolate */
  PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));

  /* Update the velocity for the next step */
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(PetscArraycpy(&v_i[npoints*dim*stage_idx],velocity,npoints*dim));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectionExplicit_RK4(
  LagRBFCtx ctx,
  DM swi, 
  DM swd, 
  const char *velocity_name
)
{
  DMSwarmCellDM  celldm;
  PetscInt       np,d,i,coor_bs,bs,v_bs,x0_bs,dim,npoints;
  PetscReal      dt;
  PetscReal      RK4_coeffs[] = {0.5, 0.5, 1.0};
  PetscReal      RK_weights[] = {1.0, 2.0, 2.0, 1.0};
  PetscReal      *x0,*coords,*velocity,*v_i;
  const char     **coorFieldName;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  PetscCall(LagRBFGetTimeStep(ctx,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));

  /* Step 1: Copy initial coordinates and initial velocity */
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
  PetscCheck(x0_bs == coor_bs,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The initial coordinates field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",coor_bs,x0_bs);
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  /* Store the initial coords */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* 
    v_i structure: [ j * npoints * dim + d * npoints + np ]
    where 
    j is the current order index, 
    npoints is the number of points, np is the current point index, 
    dim is the dimension and d is the dimension index
  */
  /* First step is u(x0), we already have it */
  PetscCall(PetscArraycpy(v_i,velocity,npoints*dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  
  /* Step 2: Interpolate u(x1) */
  PetscCall(AdvectionExplicit_SingleVelocityStage(ctx,swi,swd,RK4_coeffs[0]*dt,0,1,velocity_name));
  /* Step 3: Interpolate u(x2) */
  PetscCall(AdvectionExplicit_SingleVelocityStage(ctx,swi,swd,RK4_coeffs[1]*dt,1,2,velocity_name));
  /* Step 4: Interpolate u(x3) */
  PetscCall(AdvectionExplicit_SingleVelocityStage(ctx,swi,swd,RK4_coeffs[2]*dt,2,3,velocity_name));
  /* Step 5: Final step */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  for (np=0; np<npoints; np++) {
    PetscReal sum[] = {0.0, 0.0, 0.0};
    for (d=0; d<dim; d++) {
      for (i=0; i<ctx->adv_ctx->order; i++) {
        sum[d] += RK_weights[i] * v_i[i*npoints*dim + d*npoints + np];
      }
      coords[dim*np + d] = x0[dim*np + d] + dt/6.0 * sum[d];
    }
  }
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  /* Update MPI rank and cell id */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE));
  /* Update weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swi));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectionExplicit_DormandPrince(
  LagRBFCtx ctx,
  DM swi, 
  DM swd, 
  const char *velocity_name
)
{
  DMSwarmCellDM celldm;
  PetscInt      i,j,np,d,coor_bs,bs,x0_bs,v_bs,dim,npoints;
  PetscReal     *coords,*velocity,*x0,*v_i;
  PetscReal     dt,dt_new,max_err,fac,fac_min,fac_max;
  const char    **coorFieldName;
  PetscReal DP_coeffs[6][6] = {
    {1.0/5.0,                  0,           0,          0,           0,     0},
    {3.0/40.0,              9.0/40.0,           0,          0,           0,     0},
    {44.0/45.0,           -56.0/15.0,        32.0/9.0,          0,           0,     0},
    {19372.0/6561.0, -25360.0/2187.0,  64448.0/6561.0,   -212.0/729.0,           0,     0},
    {9017.0/3168.0,      -355.0/33.0,  46732.0/5248.0,     49.0/176.0, -5103.0/18656.0,     0},
    {35.0/384.0,               0,    500.0/1113.0,    125.0/192.0,  -2187.0/6784.0, 11.0/84.0}
  };
  PetscReal DP_weights[7] = {5179.0/57600.0, 0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(LagRBFGetTimeStep(ctx,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));

  /* Step 1: Copy initial coordinates and initial velocity */
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
  PetscCheck(x0_bs == coor_bs,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The initial coordinates field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",coor_bs,x0_bs);
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  /* Store the initial coords */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* 
    v_i structure: [ j * npoints * dim + d * npoints + np ]
    where 
    j is the current order index, 
    npoints is the number of points, np is the current point index, 
    dim is the dimension and d is the dimension index
  */
  /* First step is u(x0), we already have it */
  PetscCall(PetscArraycpy(v_i,velocity,npoints*dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  PetscCheck(v_bs == dim*ctx->adv_ctx->order,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The advection velocity field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",dim*ctx->adv_ctx->order,v_bs);
  
  for (i=0; i<6; i++) {
    //PetscPrintf(PETSC_COMM_WORLD,"i = %" PetscInt_FMT "\n",i);
    /* Access fields */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));

    PetscArrayzero(velocity,npoints*dim); // reset the velocity for the new step
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
    PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));

    /* Update points */
    PetscCall(DMSwarmMigrate(swi,PETSC_FALSE)); // update cell ids and MPI ranks
    PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

    /* Interpolate */
    PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
    /* Update the velocity for the next step */
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(PetscArraycpy(&v_i[npoints*dim*(i+1)],velocity,npoints*dim));
    PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  }
  /* Error */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  /* Error handling and adaptive time step */
  max_err   = PETSC_MIN_REAL;
  for (np=0; np<npoints; np++) {
    PetscInt  high_order_idx = 6; // last stage
    PetscReal low_v[] = {0.0, 0.0, 0.0};
    PetscReal a_tol = 1e-3, r_tol = 1e-3;
    PetscReal E,max_value,error,tol;
    /* Compute the low order velocity */
    for (i=0; i<7; i++) {
      for (d=0; d<dim; d++) {
        low_v[d] += DP_weights[i] * v_i[i*npoints*dim + d*npoints + np];
      }
    }
    E = 0.0;
    for (d=0; d<dim; d++) {
      max_value = PetscMax(PetscAbs(x0[d*npoints + np]), PetscAbs(coords[d*npoints + np]));
      /* Get tolerance */
      tol = a_tol + r_tol * max_value;
      /* Compute the error */
      error = dt * (v_i[high_order_idx*npoints*dim + d*npoints + np] - low_v[d]);
      E += (error * error) / (tol * tol);
    }
    E = PetscSqrtReal(1.0/dim * E);
    max_err = PetscMax(max_err, E);
  }
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  fac     = 0.9;
  fac_min = 1.0e-4;
  //max_err = 0.1;
  if (max_err <= 1.0) {
    /* If error is small we take the time step */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    for (np=0; np<npoints; np++) {
      for (d=0; d<dim; d++) {
        coords[dim*np + d] = x0[dim*np + d] + dt * v_i[6*npoints*dim + d*npoints + np];
      }
    }
    PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
    /* Update MPI rank and cell id */
    PetscCall(DMSwarmMigrate(swi,PETSC_TRUE));
    /* Update weights for interpolation */
    PetscCall(SwarmInterpSetWeights(ctx,swi));
    /* Compute the new time step */
    fac_max = 1.0e2;
    dt_new  = dt * PetscMin( fac_max, PetscMax( fac_min, fac * PetscPowReal( 1.0/max_err, 1.0/6.0 ) ) );
    PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Time step accepted with max error: %1.4e, new dt: %1.4e\n",PETSC_FUNCTION_NAME,max_err,dt_new);
    PetscCall(LagRBFSetTimeStep(ctx,dt_new));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    fac_max = 1.0;
    dt_new  = dt * PetscMin( fac_max, PetscMax( fac_min, fac * PetscPowReal( 1.0/max_err, 1.0/6.0 ) ) );
    PetscCall(LagRBFSetTimeStep(ctx,dt_new));
    PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Time step rejected with max error: %1.4e, new dt: %1.4e\n",PETSC_FUNCTION_NAME,max_err,dt_new);
    /* Reset coordinates and velocity to initial values */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(PetscArraycpy(coords,x0,npoints*dim));
    PetscCall(PetscArraycpy(velocity,&v_i[dim*npoints],npoints*dim)); 
    PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
    PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(AdvectionExplicit_DormandPrince(ctx,swi,swd,velocity_name)); // WARNING ABOUT X0 here
    PetscFunctionReturn(PETSC_SUCCESS);
  }
}

static PetscErrorCode PointAdvection_RungeKutta(LagRBFCtx ctx, DM swi, DM swd, const char *velocity_name)
{
  DMSwarmCellDM  celldm;
  PetscInt       d,i,j,np,dim,bs,v_bs,npoints,x0_bs,coor_bs;
  PetscReal      dt;
  PetscReal      *coords,*velocity,*x0,*v_i;
  PetscLogDouble time[2];
  const char     **coorFieldName;
  PetscReal RK_coeffs[4][4] = {
    {0.0, 0.0, 0.0, 0.0},
    {0.5, 0.0, 0.0, 0.0},
    {0.0, 0.5, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0}
  };
  PetscReal RK_weights[4] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(LagRBFGetTimeStep(ctx,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));

  /* Step 1: Copy initial coordinates and initial velocity */
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], &coor_bs, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",&x0_bs,NULL,(void **)&x0));
  PetscCheck(x0_bs == coor_bs,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The initial coordinates field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",coor_bs,x0_bs);
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  /* Store the initial coords */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* 
    v_i structure: [ j * npoints * dim + d * npoints + np ]
    where 
    j is the current order index, 
    npoints is the number of points, np is the current point index, 
    dim is the dimension and d is the dimension index
  */
  /* First step is always u(x0), we already have it */
  PetscCall(PetscArraycpy(v_i,velocity,npoints*dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  PetscCheck(v_bs == dim*ctx->adv_ctx->order,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The advection velocity field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",dim*ctx->adv_ctx->order,v_bs);
  /* Step 2: Runge-Kutta iterations */
  for (i=1; i<ctx->adv_ctx->order; i++) {
    //PetscPrintf(PETSC_COMM_WORLD,"RK step i = %" PetscInt_FMT "\n",i);
    /* Access fields */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));

    PetscArrayzero(velocity,npoints*dim); // reset the velocity for the new step
    /* Compute the step velocity */
    for (j=0; j<i; j++) {
      //PetscPrintf(PETSC_COMM_WORLD,"j = %" PetscInt_FMT "\n",j);
      for (np=0; np<npoints; np++) {
        for (d=0; d<dim; d++) {
          velocity[d*npoints + np] += RK_coeffs[i][j] * v_i[j*npoints*dim + d*npoints + np];
        }
        //PetscPrintf(PETSC_COMM_WORLD,"Point[%" PetscInt_FMT "]: u_j = [ %g, %g ]\n",np, v_i[j*npoints*dim + np],v_i[j*npoints*dim + npoints + np]);
      }
    }
    /* Compute the new coordinates */
    for (np=0; np<npoints; np++) {
      for (d=0; d<dim; d++) {
        coords[dim*np + d] = x0[dim*np + d] + dt * velocity[d*npoints + np];
      }
    }
    /* Restore fields */
    PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));

    /* Update points */
    PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
    PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

    /* Interpolate */
    PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
    /* Update the velocity for the next step */
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
    PetscCall(PetscArraycpy(&v_i[npoints*dim*i],velocity,npoints*dim));
    PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  }
  /* Final step: Advect */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  for (np=0; np<npoints; np++) {
    PetscReal sum[] = {0.0, 0.0, 0.0};
    for (d=0; d<dim; d++) {
      for (i=0; i<ctx->adv_ctx->order; i++) {
        sum[d] += RK_weights[i] * v_i[i*npoints*dim + d*npoints + np];
      }
      coords[dim*np + d] = x0[dim*np + d] + dt * sum[d];
    }
  }
  PetscCall(DMSwarmRestoreField(swi,"coor_init",NULL,NULL,(void **)&x0));
  PetscCall(DMSwarmRestoreField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));
  PetscCall(DMSwarmRestoreField(swi,velocity_name,&bs,NULL,(void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi,coorFieldName[0],NULL,NULL,(void **)&coords));
  /* Update MPI rank and cell id */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  Vec              test_vec;
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  AdvectionCtx     adv_ctx;
  DM               swarm_interp,swarm_data;
  PetscInt         k,d,nsteps,step,ndata,ninterp;
  PetscReal        dt,overlap,radius,epsilon,box_length,time;
  PetscBool        exists,found;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx"};
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  //p->output_dir
  ctx->mtype = GEODYNAMICS; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  /* Set epsilon parameter */
  box_length = 0.0;
  for (d=0; d<pu->dim; d++) {
    box_length = PetscMax(box_length,(pu->L[d] - pu->O[d]));
  }
  overlap = 1.5*box_length;
  radius  = overlap / PetscPowReal(pu->npart,1.0/pu->dim);//PetscSqrtReal(pu->npart);
  epsilon = 1.0 / radius;
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,&swarm_data));
  PetscCall(SwarmDataSetValues(swarm_data));
  /* Set all the partitions data structure */
  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));
  PetscCall(DMSwarmGetLocalSize(swarm_data,&ndata));
  /* Output data */
  //PetscCall(SwarmOutput(ctx,swarm_data,1,data_output,PETSC_FALSE));
  /* Output DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));

  /* Create interpolation points, only the DM is created, no points on it yet */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  /* Set the coordinates of the interpolation points */
  PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&ninterp));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm_interp,0,ninterp));

  /* Initial state */
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  /* Interpolate */
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_TRUE));

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  dt = 1.0;
  nsteps = 10;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  for (step=1; step<nsteps; step++) {
    PetscCall(LagRBFGetTimeStep(ctx,&dt));
    PetscCall(LagRBFGetTime(ctx,&time));
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]] dt = %g, time = %g\n",step,dt,time));
    /* Advect */
    adv_ctx->advection(ctx,swarm_interp,swarm_data);
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_TRUE));
  }
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}