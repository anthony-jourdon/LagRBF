static const char help[] = "Test to add particles running time backward followed by forward advection.\n\n";

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
#include "advection/advection_ctx.h"
#include "swarm/swarm_output.h"
//#include "advection/advection.h"

//DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)

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
  PetscReal radial_u  = 0.0;//-0.2;
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

static PetscErrorCode SwarmInterp_AddMarkersCircleFromOptions_2D(PartitionOfUnity pu, DM sw, InsertMode mode)
{
  DM         da;
  PetscInt   m[] = {0,0,1};
  PetscInt   d,i,j,k,dim,npoints;
  PetscReal  O[]={0.0, 0.0, 0.0};
  PetscReal  L[]={1.0, 1.0, 1.0};
  PetscReal  dx[3];
  PetscReal  *coor;
  PetscBool  found,spherical=PETSC_FALSE;
  const char *prefix;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetDimension(da,&dim));
  PetscCall(DMGetOptionsPrefix(sw,&prefix));
  d = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-npi",m,&d,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snpi not found. User must provide the number of interpolation points in each dimension.",prefix);
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%snpi must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-O",O,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sO must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-L",L,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sL must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);

  PetscCall(DMSwarmSetPointsSphericalCoordinates(sw,O,L,m,mode,PETSC_TRUE));
  //PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,INSERT_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  Vec              test_vec;
  LagRBFCtx        ctx;
  AdvectionCtx     adv_ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_interp,swarm_data;
  PetscInt         k,d,nsteps,step,ndata,ninterp;
  PetscReal        dt,b_dt,overlap,radius,epsilon,box_length,time;
  PetscBool        exists,found;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx"};
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  //p->output_dir
  ctx->mtype = STAGYY; // Set the model type
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
  PetscCall(SwarmOutput(ctx,swarm_data,1,data_output,PETSC_FALSE));
  /* Output DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));

  /* Create interpolation points */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  /* Set the first points, required to call at least once with INSERT_VALUES before ADD_VALUES */
  PetscCall(SwarmInterp_AddMarkersCircleFromOptions_2D(pu,swarm_interp,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  nsteps = 10; // number of time steps for advection
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  b_dt = -dt;
  PetscCall(LagRBFSetTimeStep(ctx,b_dt));
  /* First we run backward */
  for (step=nsteps-1; step>=0; step--) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,step*dt));
    /* Set step, dt, time */
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFGetTime(ctx,&time));
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    /* Add markers at "surface" and set their weights */
    //PetscCall(SwarmInterpAddMarkersFromOptions(ctx,swarm_interp));
    PetscCall(SwarmInterp_AddMarkersCircleFromOptions_2D(pu,swarm_interp,ADD_VALUES));
    /* Set the weights for the interpolation points */
    PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output (for debugging) */
    //PetscCall(SwarmOutput(ctx,swarm_interp,1,data_output,PETSC_FALSE));
    /* Advect */
    adv_ctx->advection(ctx,swarm_interp,swarm_data);
  }
  /* Now we have all our markers we give them a unique index */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&ninterp));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm_interp,0,ninterp));
  /* Now we run forward */
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  PetscCall(LagRBFGetTimeStep(ctx,&dt));
  dt = PetscAbsReal(dt); // make sure dt is positive
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  for (step=0; step<nsteps; step++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,step*dt));
    /* Set step, dt, time */
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFGetTime(ctx,&time));
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    /* Set the weights for the interpolation points */
    //PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_TRUE));
    /* Advect */
    adv_ctx->advection(ctx,swarm_interp,swarm_data);
  }
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
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