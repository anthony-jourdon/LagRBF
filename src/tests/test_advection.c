#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"
#include "swarm_utils.h"
#include "swarm_interp.h"
#include "swarm_data.h"
#include "interpolation.h"

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

static PetscErrorCode Advection_ComputeVelocity()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCheck(v_bs == dim*ctx->int_order,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"The advection velocity field should have %" PetscInt_FMT " dofs, found: %" PetscInt_FMT "",dim*ctx->int_order,v_bs);
  /* Step 2: Runge-Kutta iterations */
  for (i=1; i<ctx->int_order; i++) {
    PetscPrintf(PETSC_COMM_WORLD,"RK step i = %" PetscInt_FMT "\n",i);
    /* Access fields */
    PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(swi,velocity_name,&bs,NULL,(void **)&velocity));
    PetscCall(DMSwarmGetField(swi,"coor_init",NULL,NULL,(void **)&x0));
    PetscCall(DMSwarmGetField(swi,"v_adv",&v_bs,NULL,(void **)&v_i));

    PetscArrayzero(velocity,npoints*dim); // reset the velocity for the new step
    /* Compute the step velocity */
    for (j=0; j<i; j++) {
      PetscPrintf(PETSC_COMM_WORLD,"j = %" PetscInt_FMT "\n",j);
      for (np=0; np<npoints; np++) {
        for (d=0; d<dim; d++) {
          velocity[d*npoints + np] += RK_coeffs[i][j] * v_i[j*npoints*dim + d*npoints + np];
        }
        PetscPrintf(PETSC_COMM_WORLD,"Point[%" PetscInt_FMT "]: u_j = [ %g, %g ]\n",np, v_i[j*npoints*dim + np],v_i[j*npoints*dim + npoints + np]);
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
      for (i=0; i<ctx->int_order; i++) {
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
  DM               swarm_interp,swarm_data;
  PetscInt         k,d,nsteps,step,ndata,ninterp;
  PetscReal        dt,overlap,radius,epsilon,box_length;
  PetscBool        exists,found;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx","v_adv"};
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
  //PetscCall(SwarmOutput(ctx,swarm_data,1,data_output));
  /* Output DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));

  /* Create interpolation points, only the DM is created, no points on it yet */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  /* Set the coordinates of the interpolation points */
  PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,swarm_interp,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&ninterp));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm_interp,0,ninterp));

  /* Interpolate */
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  PetscCall(LagRBFSetStep(ctx,0));
  PetscCall(SwarmOutput(ctx,swarm_interp,3,output_fields));
  /* Advect */
  dt = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  PetscCall(LagRBFSetStep(ctx,1));
  PetscCall(LagRBFSetTime(ctx,dt));
  //PetscCall(PointAdvection_Euler_Pointwise(swarm_interp,dt,"fields"));
  PetscCall(PointAdvection_RungeKutta(ctx,swarm_interp,swarm_data,"fields"));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,3,output_fields));

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