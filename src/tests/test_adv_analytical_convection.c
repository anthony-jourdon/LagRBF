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

static double _halton_sequence(int index, int base)
{
  double f = 1.0, r = 0.0;
  while (index > 0) {
    f = f / (double)base;
    r = r + f * (double)(index % base);
    index = index / base;
  }
  return r;
}

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

static PetscErrorCode SetAnnulusCoordinates(PetscInt n[],PetscReal Radius[], PetscReal coords[])
{
  PetscInt  i,j,nidx;
  PetscReal outer_circ,annulus_width,pi,sx[2];
  PetscFunctionBeginUser;

  pi = 3.141592653589793;
  outer_circ = 2.0 * pi * Radius[1];
  annulus_width = Radius[1] - Radius[0];

  sx[0] = outer_circ / (float)n[0];
  sx[1] = annulus_width / (float)n[1];

  for (j=0; j<n[1]; j++) {
    for (i=0; i<n[0]; i++) {
      PetscReal xi,yi,t;
      nidx = i + j*n[0];
      xi = i * sx[0];
      yi = j * sx[1];

      t = xi / outer_circ * 2.0 * pi;
      coords[2*nidx]     = PetscCosReal(t)*(Radius[0]+yi);
      coords[2*nidx + 1] = PetscSinReal(t)*(Radius[0]+yi);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetAnnulusCoordinates_Halton(PetscInt npoints, PetscReal Radius[], PetscReal coords[])
{
  PetscInt  p;
  PetscReal dtheta,pi;
  PetscFunctionBeginUser;

  pi = 3.141592653589793;
  dtheta = 2.0 * pi / (float)npoints;

  for (p=0; p<npoints; p++) {
    PetscReal u,r,theta;

    theta = p * dtheta;
    u = _halton_sequence(p+1,2); // base 2 can be changed
    /*
    Scale the radius r in [R0,R1] such that the points are uniformly distributed in the annulus
    Area = pi*(R1^2 - R0^2)
    r    = sqrt( (R1^2 - R0^2)*u + R0^2 )
    */
    r = PetscSqrtReal( ( Radius[1]*Radius[1] - Radius[0]*Radius[0] )*u + Radius[0]*Radius[0] );
    coords[2*p    ] = r * PetscCosReal(theta);
    coords[2*p + 1] = r * PetscSinReal(theta);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmSetAnnulusCoordinates(DM swarm, PetscInt n[],PetscReal Radius[], PetscBool halton)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;
  PetscReal     *coords;
  const char    **coorFieldName;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Set coords */
  if (halton) {
    PetscCall(SetAnnulusCoordinates_Halton(n[0]*n[1],Radius,coords));
  } else {
    PetscCall(SetAnnulusCoordinates(n,Radius,coords));
  }
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataPoints(LagRBFCtx ctx, DM *sw)
{
  DMSwarmCellDM celldm;
  DM            swarm;
  PetscInt      d,dim,nn,npoints;
  PetscInt      m[]={120,10};
  PetscReal     Radius[]={1.0, 2.0};
  PetscReal     *coords;
  PetscBool     found;
  char          option_name[PETSC_MAX_PATH_LEN];
  const char    *prefix,**coorFieldName;
  PetscFunctionBeginUser;
  
  PetscCall(SwarmDataCreate(ctx,&swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",prefix));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,m,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,nn);
  nn = dim;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,&found));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sRadius must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  
  npoints = m[0]*m[1];
  PetscCall(DMSwarmSetLocalSizes(swarm, npoints, -1));
  PetscCall(SwarmSetAnnulusCoordinates(swarm,m,Radius,PETSC_FALSE));
  #if 0
  /* Set the coordinate field */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Set coords */
  PetscCall(SetAnnulusCoordinates(m,Radius,coords));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  #endif
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DataPointsSetVariables(DM swarm)
{
  DMSwarmCellDM celldm;
  PetscInt      p,dim,npoints,nn;
  PetscInt      ncells = 4;
  PetscReal     Radius[]={1.0,2.0};
  PetscReal     *coords,*velocity;
  const char    **coorFieldName,*prefix;
  PetscFunctionBeginUser;
  /* Set the coordinate field */
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  nn = dim;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,NULL));
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sRadius must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,dim,nn);
  PetscCall(PetscOptionsGetInt(NULL,prefix,"-ncells",&ncells,NULL));

  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "fields", NULL, NULL, (void **)&velocity));
  for (p=0; p<npoints; p++) {
    /* We will assume that the velocity is the ndim first field dof */
    PetscCall(SetAnalyticalConvectionVelocity(&coords[dim*p],Radius,ncells,&velocity[dim*p]));
  }
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  AdvectionCtx     adv_ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_interp,swarm_data;
  PetscInt         k,d,nsteps,step,ndata,nn,output_frequency=1;
  PetscInt         nip[] = {900,75};  
  PetscReal        dt,b_dt,overlap,radius,epsilon,box_length,time;
  PetscReal        Radius[] = {1.0,2.0};
  PetscBool        exists,found;
  const char       *data_output[] = {"fields"};
  const char       *output_fields[] = {"fields","pidx"};
  const char       *prefix;
  
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
    PetscCall(PartitionSetShapeFactor(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,&swarm_data));
  PetscCall(DataPointsSetVariables(swarm_data));
  /* Output data */
  PetscCall(SwarmOutput(ctx,swarm_data,1,data_output,PETSC_FALSE));
  /* Output DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));

  /* Setup RBF matrices (include Cholesky factorization) */
  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));
  
  /* Create interpolation points */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  PetscCall(DMGetOptionsPrefix(swarm_interp,&prefix));
  nn = 2;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,NULL));
  nn = 2;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-n",nip,&nn,&found));
  PetscCall(DMSwarmSetLocalSizes(swarm_interp, nip[0]*nip[1], -1));
  PetscCall(SwarmSetAnnulusCoordinates(swarm_interp,nip,Radius,PETSC_TRUE));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm_interp,0,nip[0]*nip[1]));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-output_frequency",&output_frequency,NULL));
  nsteps = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  dt = 1.0e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  for (step=0; step<nsteps; step++) {
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,time));
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    if (step % output_frequency == 0) {
      PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_TRUE));
    }
    /* Advect */
    adv_ctx->advection(ctx,swarm_interp,swarm_data);
    LagRBFGetTimeStep(ctx,&dt);  
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
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