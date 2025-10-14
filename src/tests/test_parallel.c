static const char help[] = "Testing functionalities in parallel.\n\n";

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

static PetscErrorCode SetAnalyticalConvectionVelocity(PetscReal coor[], PetscReal Radius[], PetscInt ncells, PetscReal velocity[])
{
  PetscReal x,y,r,theta,A,B,R1,R2;
  PetscReal f,g,vr,vtheta;
  PetscFunctionBeginUser;

  x  = coor[0];   y  = coor[1];
  R1 = Radius[0]; R2 = Radius[1];
  
  r = PetscSqrtReal(x*x + y*y);
  theta = PetscAtan2Real(y,x);
  
  A = 2.0*( PetscLogReal(R1) - PetscLogReal(R2) ) / ( R2*R2*PetscLogReal(R1) - R1*R1*PetscLogReal(R2) );
  B = ( R2*R2-R1*R1 )/( R2*R2*PetscLogReal(R1) - R1*R1*PetscLogReal(R2) );
    
  f = A*r+B/r;
  g = 0.5*A*r + B/r*PetscLogReal(r) - 1.0/r;

  vr     = ncells * g * PetscSinReal(ncells * theta);
  vtheta = f * PetscCosReal(ncells * theta);
  
  velocity[0] = vr*PetscCosReal(theta) - vtheta*PetscSinReal(theta);
  velocity[1] = vr*PetscSinReal(theta) + vtheta*PetscCosReal(theta);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode circle_gaussian(PetscReal coor[], PetscReal r, PetscReal *value)
{
  PetscReal x,y;
  PetscFunctionBeginUser;
  x = coor[0]; y = coor[1];
  *value = 0.5*PetscExpReal(-3.0*pow(PetscSqrtReal(x*x + y*y) - r, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  Create the swarm data.
  Brute force: 
    COMM_SELF i.e., sequential DMSwarm
    All data points are created on all processes.
    Each process will then select the data points that belong to its partition.
    This is not efficient, but it is simple because no need to identify the ranks and elements where data points should
    be copied.
*/
static PetscErrorCode CreateSwarmData_BruteForce(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               das,swarm;
  PetscInt         dim;
  int              rank;
  char             name[PETSC_MAX_PATH_LEN],dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)das), &swarm)); // COMM_SELF
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, das));
  PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "swarm_data");
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "%s_%d",name,rank);
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"rbf",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  PetscCall(DMSetUp(swarm));

  *sw = swarm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmDataSetValues(DM swarm, PetscReal Radius[], PetscInt ncells)
{
  DMSwarmCellDM celldm;
  PetscInt      p,dim,npoints,bs;
  PetscReal     *coords,*fields;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "fields", &bs, NULL, (void **)&fields));
  for (p=0; p<npoints; p++) {
    /* We will assume that the velocity is the ndim first field dof */
    PetscCall(SetAnalyticalConvectionVelocity(&coords[dim*p],Radius,ncells,&fields[bs*p]));
    PetscCall(circle_gaussian(&coords[dim*p],1.5,&fields[bs*p + 2]));
  }
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&fields));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarmData(LagRBFCtx ctx, DM *sw)
{
  DM         swarm=NULL;
  PetscInt   d,dim,nn,ncells;
  PetscInt   m[]={120,10,1};
  PetscReal  min[] = {0.0,0.0,0.0}, max[] = {1.0,6.3,3.15};
  PetscReal  Radius[]={1.0, 2.0};
  PetscReal  pi = 3.141592653589793;
  PetscBool  found;
  const char *prefix;
  char       option_name[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(CreateSwarmData_BruteForce(ctx,&swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",prefix));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,m,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,nn);
  nn = dim;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sRadius must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  
  ncells = 4;
  PetscCall(PetscOptionsGetInt(NULL,prefix,"-ncells",&ncells,NULL));
  
  min[0] = Radius[0]; max[0] = Radius[1];
  min[1] = 0.0;       max[1] = 2.0*pi;
  PetscCall(DMSwarmSetPointsSphericalCoordinates(swarm,min,max,m,INSERT_VALUES,PETSC_TRUE));
  PetscCall(SwarmDataSetValues(swarm,Radius,ncells));
  *sw = swarm;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Not optimized for parallel i.e., every rank does the same work... */
static PetscErrorCode CreateSwarmInterp(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               dap,swarm;
  PetscInt         d,dim,nn,npoints,npoints_i;
  PetscInt         nip[] = {20,20,1};
  PetscReal        Radius[]={1.0, 2.0};
  PetscReal        min[] = {0.0,0.0,0.0}, max[] = {1.0,6.3,3.15};
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  /* Create interpolation points */
   PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dap), &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, dap));
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  /* Unique point index across all ranks */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pidx",1,PETSC_INT));
  /* Weights for interpolation */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",pu->npart,PETSC_REAL));
  /* Fields to interpolate, also contain the velocity even in advection mode */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  /* If advection is required, special fields are added */
  if (ctx->advection) {
    PetscCall(SwarmInterpCreateAdvectionFields(ctx, swarm));
  }
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));

  nn = 2;
  PetscCall(PetscOptionsGetRealArray(NULL,dm_prefix,"-Radius",Radius,&nn,NULL));
  nn = 2;
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-n",nip,&nn,NULL));
  npoints = 1;
  for (d=0; d<dim; d++) { npoints *= nip[d]; }
  /* The following automatically removes the points not living inside the DMCell */
  min[0] = Radius[0]; max[0] = Radius[1];
  PetscCall(DMSwarmSetPointsSphericalCoordinates(swarm, min, max, nip, INSERT_VALUES, PETSC_TRUE));
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints_i)); // rank local points
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm,0,npoints_i));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  Partition        *partitions;
  CellInfo         *ci;
  DM               das,swarm_data,swarm_interp;
  PetscInt         k,d,nsteps,step,global_npart,output_frequency=1;
  PetscInt         nel,nen;
  PetscInt         M[3];
  const PetscInt   *elidx;
  PetscReal        overlap,radius,epsilon,box_length,dt,time;
  PetscBool        exists;
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
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  /* Set epsilon */
  PetscCall(PUGetEpsilon(pu,&epsilon));
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Create data points */
  PetscCall(CreateSwarmData(ctx, &ctx->swd));
  PetscCall(LagRBFGetSwarmData(ctx,&swarm_data));
  PetscCall(SwarmOutput(ctx, swarm_data, 1, data_output, PETSC_FALSE));
  /* Factor RBF matrix */
  PetscCall(PartitionSetUp_All(partitions, pu->npart, swarm_data));
  /* Set up cell information */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  PetscCall(CellInfoSetUp(pu,ci,swarm_data));

  PetscCall(CreateSwarmInterp(ctx, &ctx->swi));
  PetscCall(LagRBFGetSwarmInterp(ctx,&swarm_interp));
  /* Interpolate */
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));

  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));
  
  /* Output parallel DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));
  //PetscCall(PartitionOfUnityMeshView_Seq(pu));
  /* Free memory */
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_advection()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  CellInfo         *ci;
  Partition        *partitions;
  DM               das,swarm_data,swarm_interp;
  PetscInt         k,d,nsteps,step,global_npart,output_frequency=1;
  PetscInt         nel,nen;
  const PetscInt   *elidx;
  PetscInt         M[3];
  PetscReal        overlap,radius,epsilon,box_length,dt,time;
  PetscBool        exists;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx","velocity"};
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  //p->output_dir
  ctx->mtype = GEODYNAMICS; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  DMDAGetInfo(pu->dap,NULL,&M[0],&M[1],&M[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  global_npart = 1;
  for (d=0; d<pu->dim; d++) global_npart *= M[d];
  /* Set epsilon parameter */
  box_length = 0.0;
  for (d=0; d<pu->dim; d++) {
    box_length = PetscMax(box_length,(pu->L[d] - pu->O[d]));
  }
  overlap = 1.5*box_length;
  radius  = overlap / PetscPowReal(global_npart,1.0/pu->dim);
  epsilon = 1.0 / radius;
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  //PetscCall(PrintDMDAInfo(ctx));
  //PetscCall(PrintPartitionMap(ctx));
  PetscCall(CreateSwarmData(ctx, &swarm_data));
  PetscCall(SwarmOutput(ctx, swarm_data, 1, data_output, PETSC_FALSE));

  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  PetscCall(CellInfoSetUp(pu,ci,swarm_data));

  PetscCall(PartitionSetUp_All(partitions, pu->npart, swarm_data));

  PetscCall(CreateSwarmInterp(ctx, &swarm_interp));
  /* Interpolate */
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Copy the first dim fields into "velocity" */
  {
    PetscInt  npoints;
    PetscReal *interp_fields,*velocity;
    PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints));
    PetscCall(DMSwarmGetField(swarm_interp, "fields", NULL, NULL, (void **)&interp_fields));
    PetscCall(DMSwarmGetField(swarm_interp, "velocity", NULL, NULL, (void **)&velocity));
    //PetscCall(CopyContiguousBlockData_Real(velocity,interp_fields,npoints,pu->dim,ctx->n_ifields,0));
    PetscCall(CopyContiguousBlockData(velocity, interp_fields, npoints, pu->dim, ctx->n_ifields, 0));
    PetscCall(DMSwarmRestoreField(swarm_interp, "fields", NULL, NULL, (void **)&interp_fields));
    PetscCall(DMSwarmRestoreField(swarm_interp, "velocity", NULL, NULL, (void **)&velocity));
  }
  PetscCall(SwarmOutput(ctx,swarm_interp,3,output_fields,PETSC_FALSE));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-output_frequency",&output_frequency,NULL));
  nsteps = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  dt = 1.0e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  for (step=1; step<nsteps; step++) {
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,time));
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Copy the first dim fields into "velocity" */
    {
      PetscInt  npoints;
      PetscReal *interp_fields,*velocity;
      PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints));
      PetscCall(DMSwarmGetField(swarm_interp, "fields", NULL, NULL, (void **)&interp_fields));
      PetscCall(DMSwarmGetField(swarm_interp, "velocity", NULL, NULL, (void **)&velocity));
      //PetscCall(CopyContiguousBlockData_Real(velocity, interp_fields,npoints,pu->dim,ctx->n_ifields,0));
      /* Need to check that on many points and many time steps */
      PetscCall(CopyContiguousBlockData(velocity, interp_fields, npoints, pu->dim, ctx->n_ifields, 0));
      PetscCall(DMSwarmRestoreField(swarm_interp, "fields", NULL, NULL, (void **)&interp_fields));
      PetscCall(DMSwarmRestoreField(swarm_interp, "velocity", NULL, NULL, (void **)&velocity));
    }
    /* Output */
    if (step % output_frequency == 0) {
      PetscCall(SwarmOutput(ctx,swarm_interp,3,output_fields,PETSC_FALSE));
    }
    /* Advect */
    //adv_ctx->advection(ctx,swarm_interp,swarm_data);
    PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
    LagRBFGetTimeStep(ctx,&dt);  
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
  }
  /* Free memory */
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
  //PetscCall(Test_advection());

  PetscCall(PetscFinalize());
  
  return 0;
}