static const char help[] = "Testing advection schemes.\n\n";

#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "partitions/partition_solve.h"
#include "cell_info.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "interpolation.h"
#include "advection/advection_ctx.h"
#include "swarm/swarm_output.h"

#if 0
typedef struct {
  PetscReal Radius[2];
  PetscInt  ncells;
} ConvectionCtx;

static PetscErrorCode AnalyticalRotationVelocity_2D(PetscReal coor[], PetscReal velocity[])
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

static PetscErrorCode AnalyticalRotationVelocity_3D(PetscReal coor[], PetscReal velocity[])
{
  PetscReal r,theta,phi;
  PetscReal angular_u = 0.5;
  PetscFunctionBeginUser;
  r     = PetscSqrtReal(coor[0]*coor[0] + coor[1]*coor[1] + coor[2]*coor[2]);
  theta = PetscAtan2Real(PetscSqrtReal(coor[0]*coor[0] + coor[1]*coor[1]), coor[2]);
  phi   = PetscAtan2Real(coor[1], coor[0]);
  velocity[0] = -angular_u * r * PetscSinReal(theta) * PetscSinReal(phi);
  velocity[1] = angular_u * r * PetscSinReal(theta) * PetscCosReal(phi);
  velocity[2] = 0.0;  // No vertical rotation
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AnalyticalRotationVelocity(PetscReal coor[], PetscReal velocity[], PetscInt dim)
{
  PetscFunctionBeginUser;
  switch (dim) {
  case 2:
    AnalyticalRotationVelocity_2D(coor, velocity);
    break;
  case 3:
    AnalyticalRotationVelocity_3D(coor, velocity);
    break;
  
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AnalyticalConvectionVelocity(PetscReal coor[], PetscReal Radius[], PetscInt ncells, PetscReal velocity[])
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

PetscErrorCode PT_AnalyticalConvectionVelocity(PetscReal coor[], PetscReal velocity[], void *ctx)
{
  ConvectionCtx *user = (ConvectionCtx*)ctx;
  PetscFunctionBeginUser;
  PetscCall(AnalyticalConvectionVelocity(coor,user->Radius,user->ncells,velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PT_AnalyticalRotationVelocity(PetscReal coor[], PetscReal velocity[], void *ctx)
{
  PetscInt dim = *((PetscInt*)ctx);
  PetscFunctionBeginUser;
  PetscCall(AnalyticalRotationVelocity(coor,velocity,dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAdvectionSetVelocityFunction(AdvectionCtx ctx, PetscBool convection, ConvectionCtx *conv_ctx, PetscInt *dim)
{
  PetscFunctionBeginUser;
  if (convection) {
    PetscCheck(*dim == 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Convection analytical velocity is only implemented in 2D.");
    PetscCall(AdvectionSetVelocityFunction(ctx,PT_AnalyticalConvectionVelocity,(void *)conv_ctx));
  } else {
    PetscCall(AdvectionSetVelocityFunction(ctx,PT_AnalyticalRotationVelocity,(void *)dim));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAdvection_SetAdvectionMode(AdvectionCtx ctx, ConvectionCtx *conv_ctx, PetscInt *dim)
{
  PetscBool     analytical = PETSC_FALSE;
  PetscBool     convection = PETSC_FALSE;
  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-advection_analytical",&analytical,NULL));
  if (analytical) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ ADVECTION ]]: Using analytical function.\n");
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-swarm_data_convection",&convection,NULL));
    PetscCall(TestAdvectionSetVelocityFunction(ctx,convection,conv_ctx,dim));
  } else { /* default, nothing to do */
    PetscPrintf(PETSC_COMM_WORLD,"[[ ADVECTION ]]: Using interpolation from RBF.\n");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode SwarmDataSetValues(DM swarm, PetscReal Radius[], PetscInt ncells, PetscBool convection)
{
  DMSwarmCellDM celldm;
  PetscInt      p,dim,npoints,bs;
  PetscReal     *coords,*fields;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMGetDimension(swarm,&dim));
  if (convection) { PetscCheck(dim == 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Convection analytical velocity is only implemented in 2D."); }
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "fields", &bs, NULL, (void **)&fields));
  for (p=0; p<npoints; p++) {
    /* We will assume that the velocity is the ndim first field dof */
    if (convection) {
      PetscCall(AnalyticalConvectionVelocity(&coords[dim*p],Radius,ncells,&fields[bs*p]));
    } else {
      PetscCall(AnalyticalRotationVelocity(&coords[dim*p], &fields[bs*p], dim));
    }
  }
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&fields));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarmData(LagRBFCtx ctx, DM *sw)
{
  DM         swarm=NULL;
  PetscInt   dim,nn,ncells;
  PetscInt   m[]={120,10,1};
  PetscReal  Radius[]={1.0, 2.0};
  PetscReal  min[] = {0.0,0.0,0.0}, max[] = {1.0,6.3,3.15};
  PetscBool  found,convection;
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
  nn = 2;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,&found));
  PetscCheck(nn == 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sRadius must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  ncells = 4;
  PetscCall(PetscOptionsGetInt(NULL,prefix,"-ncells",&ncells,NULL));
  convection = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,prefix,"-convection",&convection,NULL));

  /* The following automatically removes the points not living inside the DMCell */
  min[0] = Radius[0]; max[0] = Radius[1];
  PetscCall(DMSwarmSetPointsSphericalCoordinates(swarm, min, max, m, INSERT_VALUES, PETSC_TRUE));
  PetscCall(SwarmDataSetValues(swarm,Radius,ncells,convection));
  *sw = swarm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarmInterp(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               dap,swarm;
  PetscInt         d,dim,nn,npoints,npoints_i;
  PetscInt         nip[] = {20,20,1};
  PetscReal        Radius[]={1.0, 2.0};
  PetscReal        min[] = {0.0,0.0,0.0}, max[] = {1.0,6.3,3.15};
  const char       *dm_prefix;
  PetscFunctionBeginUser;
  /* Create interpolation points */
  PetscCall(SwarmInterpCreate(ctx,&swarm));
  PetscCall(DMGetOptionsPrefix(swarm,&dm_prefix));
  PetscCall(DMGetDimension(swarm,&dim));
  nn = 2;
  PetscCall(PetscOptionsGetRealArray(NULL,dm_prefix,"-Radius",Radius,&nn,NULL));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-n",nip,&nn,NULL));
  /* The following automatically removes the points not living inside the DMCell */
  min[0] = Radius[0]; max[0] = Radius[1];
  PetscCall(DMSwarmSetPointsSphericalCoordinates(swarm, min, max, nip, INSERT_VALUES, PETSC_TRUE));

  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints_i)); // rank local points
  PetscCall(SwarmFieldAssignUniqueIndex(swarm,DMSwarmField_pid,0,npoints_i));
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CountZerosInRBFMatrix(LagRBFCtx ctx, DM swarm)
{
  DMSwarmCellDM    celldm;
  PartitionOfUnity pu;
  Partition       *partitions;
  PetscInt         n,dim,npart;
  const char       **coordFieldsName;
  PetscReal        *coords;
  PetscReal        epsilon;
  PetscMPIInt      rank;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscCall(PUGetSupportFactor(pu,&epsilon));
  
  PetscCall(DMSwarmSortGetAccess(swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFieldsName));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm, coordFieldsName[0], NULL, NULL, (void **)&coords));

  for (n=0; n<npart; n++) {
    Partition  p = partitions[n];
    PetscInt64 zero_count = 0;
    PetscInt   npoints,i,j,d;
    PetscInt   *pidx;
    PetscReal  ratio;

    PetscCall(PartitionFillDataPointsIdx(p,swarm));
    PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
    if (npoints == 0) continue;
    /* RBF matrix */
    for (i=0; i<npoints; i++) {
      for (j=i; j<npoints; j++) {
        PetscReal dist = 0.0;
        /* Compute the distance between points */
        for (d=0; d<dim; d++) {
          dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
        }
        dist = PetscSqrtReal(dist);
        if (1.0 - epsilon * dist < 0.0) {
          zero_count++;
        }
      }
    }
    ratio = (PetscReal)(2*zero_count) / (PetscReal)(npoints*npoints);
    PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Partition[%" PetscInt_FMT "] has %" PetscInt_FMT " data points and %" PetscInt64_FMT " zero entries = %g ratio, in the RBF matrix\n",rank,n,npoints,2*zero_count,ratio);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_advection()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  ConvectionCtx    conv_ctx;
  CellInfo         *cellinfo;
  Partition        *partitions;
  DM               swarm_data,swarm_interp;
  PetscInt         M[3],d,k,global_npart;
  PetscInt         step,output_frequency,ncells,nn;
  PetscReal        box_length,overlap,radius,epsilon;
  PetscReal        dt,time,time_final,output_dt,time_last_output;
  PetscReal        Radius[] = {1.0,2.0};
  PetscBool        exists;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"DMSwarm_pid","fields","velocity","DMSwarm_rank"}; //"pidx"
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  //p->output_dir
  ctx->mtype = STAGYY; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&cellinfo));

  PetscCall(PartitionOfUnityMeshView(pu));
  PetscCall(PartitionOfUnityMeshView_Seq(pu));

  PetscCall(PUGetSupportFactor(pu,&epsilon));
  PetscPrintf(PETSC_COMM_WORLD,"Epsilon from PU: %g\n",epsilon);
  PetscCall(CreateSwarmData(ctx, &ctx->swd));
  swarm_data = ctx->swd;
  PetscCall(SwarmOutput(ctx, swarm_data, 1, data_output, PETSC_FALSE));

  /* Set advection mode */
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  nn = 2;
  PetscOptionsGetRealArray(NULL,NULL,"-swarm_data_Radius",Radius,&nn,NULL);
  PetscOptionsGetInt(NULL,NULL,"-swarm_data_ncells",&ncells,NULL);
  conv_ctx.ncells = ncells;
  conv_ctx.Radius[0] = Radius[0];
  conv_ctx.Radius[1] = Radius[1];
  PetscCall(TestAdvection_SetAdvectionMode(adv_ctx, &conv_ctx, &pu->dim));

  /* Set up cell information */
  PetscCall(CellInfoSetUp(pu,cellinfo,swarm_data));

  PetscCall(CreateSwarmInterp(ctx, &ctx->swi));
  swarm_interp = ctx->swi;
  PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
  /* Count zeros in RBF matrix */
  //PetscCall(CountZerosInRBFMatrix(ctx,swarm_data));
  PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
#if 1
  
  /* Interpolate */
  //PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Copy the first dim fields into "velocity" */
  PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
  output_frequency = 100;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-output_frequency",&output_frequency,NULL));
  time_last_output = 0.0;
  output_dt = 0.1;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-output_dt",&output_dt,NULL));
  dt = 1.0e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  time_final = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_final",&time_final,NULL));
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  step = 0;
  while (time < time_final) {
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,time));
    /* Interpolate */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    //PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
    /* Output */
    if (step % output_frequency == 0) {
      PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
    }
    /* Output every given dt_output */
    if (output_dt > 0.0) {
      if (time - time_last_output >= output_dt - 1e-10) {
        PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
        time_last_output = time;
      }
    }
    /* Advect */
    PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
    PetscCall(LagRBFGetTimeStep(ctx,&dt));
    if (time + dt > time_final) {
      dt = time_final - time;
      PetscCall(LagRBFSetTimeStep(ctx,dt));
    }
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(LagRBF_PartitionsReset(ctx));
    step++;
  }
#endif
  /* Free memory */
  //PetscCall(DMDestroy(&swarm_interp));
  //PetscCall(DMDestroy(&swarm_data));

  PetscCall(LagRBFCtxDestroy(&ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode RigidRotationVelocity_3D(PetscReal coords[], PetscReal velocity[], void *ctx)
{
  PetscReal *angular_u = (PetscReal*)ctx;
  PetscReal r,theta,phi;
  PetscFunctionBeginUser;

  r = PetscSqrtReal(coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2]);
  theta = PetscAtan2Real(PetscSqrtReal(coords[0]*coords[0] + coords[1]*coords[1]), coords[2]);
  phi   = PetscAtan2Real(coords[1], coords[0]);
  
  velocity[0] = angular_u[1] * r * PetscCosReal(theta) - angular_u[2] * r * PetscSinReal(theta) * PetscSinReal(phi);
  velocity[1] = angular_u[2] * r * PetscSinReal(theta) * PetscCosReal(phi) - angular_u[0] * r * PetscCosReal(theta);
  velocity[2] = angular_u[0] * r * PetscSinReal(theta) * PetscSinReal(phi) - angular_u[1] * r * PetscSinReal(theta) * PetscCosReal(phi);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RegisterRigidRotationVelocity(AdvectionCtx ctx, PetscReal *angular_u)
{
  PetscInt  nn,d;
  PetscReal angular_vel;
  PetscReal rotation_vec[] = {0.0, 0.0, 1.0}; // Default rotation around z-axis
  PetscBool found;
  PetscFunctionBeginUser;

  angular_vel = 0.5;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-angular_velocity",&angular_vel,NULL));
  nn = 3;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-rotation_vector",rotation_vec,&nn,&found));
  if (found) { PetscCheck(nn == 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -rotation_vector must have 3 values, found: %" PetscInt_FMT "",nn); }

  for (d=0; d<3; d++) { angular_u[d] = angular_vel * rotation_vec[d]; }
  PetscCall(AdvectionSetVelocityFunction(ctx,RigidRotationVelocity_3D,(void *)angular_u));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_RigidRotation3D()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_interp;
  PetscInt         step,npoints_i,max_steps;
  PetscReal        dt, time_final, time;
  PetscReal        angular_u[3];
  PetscFunctionBeginUser;

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  /* Create the output directory */
  PetscCall(CreateDirectory(ctx->output_dir));
  
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Advection context */
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(RegisterRigidRotationVelocity(adv_ctx, angular_u));

  /* Create the swarm */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  PetscCall(DMSwarmSetPointsCoordinates_FromOptions(swarm_interp,INSERT_VALUES));
  /* Set weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  
  /* Timestep */
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  PetscCall(LagRBFSetTimeStep(ctx,dt));

  /* Time limit */
  time_final = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_final",&time_final,NULL));
  time = 0.0;
  step = 0;
  PetscCall(LagRBFSetStep(ctx,step));
  PetscCall(LagRBFSetTime(ctx,time));
  max_steps = (PetscInt)(time_final/dt);

  /* Initialize velocity */
  PetscCall(AdvectionInitializeTimestepVelocity(ctx, swarm_interp));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));

  for (step=0; step<max_steps; step++) {
    PetscPrintf(PETSC_COMM_WORLD,"    Advection step [[ %" PetscInt_FMT " ]] : time = %1.4e, dt = %1.4e\n",step,time,dt);
    /* Advect */
    PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,NULL));
    /* Initialize velocity for next timestep */
    PetscCall(AdvectionInitializeTimestepVelocity(ctx, swarm_interp));
    /* Update time */
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    PetscCall(LagRBFSetStep(ctx,step));
  }
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));
  /* Free memory */
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,help));

  //PetscCall(Test_advection());
  PetscCall(Test_RigidRotation3D());

  PetscCall(PetscFinalize());
  
  return 0;
}