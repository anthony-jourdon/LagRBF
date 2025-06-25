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

#include <immintrin.h>


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

static PetscReal velocity_function(PetscReal x, PetscReal y, PetscInt dof)
{
  PetscReal r,theta;
  PetscReal u[2];
  PetscFunctionBeginUser;
  r     = PetscSqrtReal(x*x + y*y);
  theta = PetscAtan2Real(y, x);
  u[0] = -0.5 * r * PetscSinReal(theta) * PetscCosReal(y);
  u[1] =  0.5 * r * PetscCosReal(theta) * PetscCosReal(x); 
  PetscFunctionReturn(u[ dof ]); // return the magnitude of the velocity
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
  PetscInt      p,d,bs,dim,npoints;
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
    /* We will assume that the velocity is ALWAYS the ndim first field dof */
    for (d=0; d<dim; d++) {
      fields[bs*p + d] = velocity_function(coords[dim*p], coords[dim*p+1], d);
    }
    fields[bs*p + dim  ] = triple_gaussians(coords[dim*p], coords[dim*p+1]); // first field
    fields[bs*p + dim+1] = circle_gaussian(coords[dim*p], coords[dim*p+1]); // second field
    fields[bs*p + dim+2] = discontinuous_cross_planes(coords[dim*p], coords[dim*p+1]); // third field
  }
  PetscCall(DMSwarmRestoreField(sw,"fields",&bs,NULL,(void **)&fields));
  PetscCall(DMSwarmRestoreField(sw,coorFieldName[0],NULL,NULL,(void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmUpdateCellId(DM sw)
{
  DMSwarmCellDM       celldm;
  PetscSF             sfcell = NULL;
  const PetscSFNode   *LA_sfcell;
  DMPointLocationType ltype;
  DM                  da;
  Vec                 coords;
  PetscInt            p,npoints;
  PetscInt            *cell_idx;
  const char          **coorFieldName,*cellid;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellid));
  PetscCall(DMSwarmCellDMGetDM(celldm,&da));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm,NULL,&coorFieldName));
  PetscCall(DMSwarmCreateLocalVectorFromField(sw,coorFieldName[0],&coords));
  /* locate points */
  ltype = DM_POINTLOCATION_NEAREST;
  PetscCall(DMLocatePoints(da,coords,ltype,&sfcell));
  PetscCall(PetscSFGetGraph(sfcell,NULL,NULL,NULL,&LA_sfcell));
  /* Update cell id */
  PetscCall(DMSwarmGetField(sw,cellid,NULL,NULL,(void **)&cell_idx));
  for (p=0; p<npoints; p++) {
    cell_idx[p] = LA_sfcell[p].index; // update the cell index
    //PetscPrintf(PETSC_COMM_WORLD,"point[%" PetscInt_FMT "]: cell index: %" PetscInt_FMT "\n", p, cell_idx[p]);
  }
  /* Restore */
  PetscCall(DMSwarmRestoreField(sw,cellid,NULL,NULL,(void **)&cell_idx));
  PetscCall(DMSwarmDestroyLocalVectorFromField(sw,coorFieldName[0],&coords));
  PetscCall(PetscSFDestroy(&sfcell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointAdvection_Euler_Pointwise(DM sw, PetscReal dt)
{
  DMSwarmCellDM  celldm;
  PetscInt       p,dim,bs,npoints;
  PetscReal      *coords,*fields;
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
  PetscCall(DMSwarmGetField(sw,"fields",&bs,NULL,(void **)&fields));
  for (p=0; p<npoints; p++) {
    PetscInt d;
    // the velocity is ALWAYS the first dim fields
    for (d=0; d<dim; d++) {
      coords[dim*p + d] = coords[dim*p + d] + fields[bs*p + d] * dt;
    }
  }
  PetscCall(DMSwarmRestoreField(sw,"fields",&bs,NULL,(void **)&fields));
  PetscCall(DMSwarmRestoreField(sw,coorFieldName[0],NULL,NULL,(void **)&coords));
  /*
  In DMSwarmSetPointCoordinates() the points are re-located with:
    DMLocatePoints(dm, pos, DM_POINTLOCATION_NONE, &sfcell)
  and if a point index is not found in the DM it gets set to DM_POINTLOCATION_NONE
  which is then used to ignore those points.
  This means that if a point leaved the DMDA (partitions) it will be removed
  */
  //PetscCall(DMSwarmSetPointCoordinates(sw,npoints,coords_new,PETSC_FALSE,INSERT_VALUES));
  //PetscCall(SwarmUpdateCellId(sw));
  //PetscCall(PetscFree(coords_new));
  PetscCall(DMSwarmMigrate(sw,PETSC_TRUE));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointAdvection_Euler_Vec(DM sw, PetscReal dt)
{
  DMSwarmCellDM     celldm;
  DM                da;
  Vec               coords,velocity;
  //PetscSF           sfcell = NULL;
  //const PetscSFNode *LA_sfcell;
  PetscInt          dim,npoints;
  PetscLogDouble    time[2];
  const char        **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(DMGetDimension(sw,&dim));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &da));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmCreateLocalVectorFromField(sw,coorFieldName[0],&coords));
  PetscCall(DMSwarmCreateLocalVectorFromField(sw,"velocity",&velocity));

  PetscCall(VecAXPY(coords,dt,velocity)); // coords = coords + dt * velocity
  //PetscCall(DMLocatePoints(da,coords,DM_POINTLOCATION_NONE,&sfcell));
  //PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));

  PetscCall(DMSwarmDestroyLocalVectorFromField(sw,coorFieldName[0],&coords));
  PetscCall(DMSwarmDestroyLocalVectorFromField(sw,"velocity",&velocity));
  PetscCall(PetscTime(&time[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_interp,swarm_data;
  PetscInt         k,d,nsteps,step;
  PetscReal        dt,overlap,radius,epsilon,box_length;
  PetscBool        exists;
  const char       output_dir[] = "output";
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx"};
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }
  
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
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
  /* Set interpolation points */
  PetscCall(SwarmInterpSetUp(ctx,&swarm_interp));
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,&swarm_data));
  PetscCall(SwarmDataSetValues(swarm_data));
  PetscCall(SwarmOutput(ctx,swarm_data,1,data_output));

  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields));
  PetscCall(PartitionOfUnityMeshView(pu));

  nsteps = 10; // number of time steps for advection
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  for (step=1; step<nsteps; step++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,step*dt));
    ctx->step = step;
    ctx->time = step*dt;
    /* Advection step */
    PetscCall(PointAdvection_Euler_Pointwise(swarm_interp,dt)); // update coords and cell idx
    PetscCall(SwarmInterpSetWeights(ctx,swarm_interp)); // update weights for interpolation
    /* Interpolate the data points */
    PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields));
  }
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  //PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(LagRBFInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}