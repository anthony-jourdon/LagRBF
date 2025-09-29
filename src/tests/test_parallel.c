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

static double halton_sequence(int index, int base)
{
  double f = 1.0, r = 0.0;
  while (index > 0) {
    f = f / (double)base;
    r = r + f * (double)(index % base);
    index = index / base;
  }
  return r;
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
    u = halton_sequence(p+1,2); // base 2 can be changed
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
  PetscInt   d,dim,nn,npoints,ncells;
  PetscInt   m[]={120,10,1};
  PetscReal  Radius[]={1.0, 2.0};
  PetscReal  *coords;
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

  npoints = 1;
  for (d=0; d<dim; d++) { npoints *= m[d]; }
  PetscCall(PetscMalloc1(npoints*dim,&coords));
  PetscCall(SetAnnulusCoordinates(m,Radius,coords));
  /* The following automatically removes the points not living inside the DMCell */
  PetscCall(DMSwarmSetPointCoordinates(swarm,npoints,coords,PETSC_FALSE,INSERT_VALUES));
  PetscCall(SwarmDataSetValues(swarm,Radius,ncells));
  *sw = swarm;
  PetscCall(PetscFree(coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Not optimized for parallel i.e., every rank does the same work... */
static PetscErrorCode CreateSwarmInterp(LagRBFCtx ctx, DM *sw)
{
  DM          swarm;
  PetscInt    d,nn,dim,npoints,npoints_i;
  PetscInt    nip[] = {100,100,1};
  PetscReal   Radius[] = {1.0,2.0};
  PetscReal   *coords;
  const char  *prefix;
  PetscFunctionBeginUser;
  PetscCall(SwarmInterpCreate(ctx, &swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  /* Add points */
  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  nn = 2;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-Radius",Radius,&nn,NULL));
  nn = 2;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-n",nip,&nn,NULL));
  npoints = 1;
  for (d=0; d<dim; d++) { npoints *= nip[d]; }

  PetscCall(PetscMalloc1(npoints*dim,&coords));
  PetscCall(SetAnnulusCoordinates_Halton(npoints,Radius,coords));
  PetscCall(DMSwarmSetPointCoordinates(swarm,npoints,coords,PETSC_FALSE,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints_i)); // rank local points
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm,0,npoints_i));
  *sw = swarm;
  PetscCall(PetscFree(coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmPrint(LagRBFCtx ctx, DM swarm)
{
  PartitionOfUnity pu;
  DM            da_local;
  DMSwarmCellDM celldm;
  PetscInt      np,npoints,global_npoints,nel,nen,bs;
  PetscInt      *cellid,*gtol_elidx;
  const PetscInt *elidx;
  PetscReal     *coords,*weights,*fields;
  PetscMPIInt   rank,size;
  MPI_Comm      comm;
  const char    **coordFieldName;
  const char    *cellFieldName;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetComm((PetscObject)swarm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));

  PetscCall(DMSwarmGetLocalSize(swarm,&npoints)); // rank local points
  PetscCall(DMSwarmGetSize(swarm,&global_npoints)); // global points

  PetscPrintf(
    PETSC_COMM_SELF,
    "[[ DMSwarmPrint ]], Rank: %d, local points: %" PetscInt_FMT ", global points: %" PetscInt_FMT "\n",
    rank,npoints,global_npoints
  );

  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm,NULL,&coordFieldName));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellFieldName));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm, coordFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmGetField(swarm, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swarm, "fields",&bs,NULL,(void **)&fields));

  PetscCall(PUGetDMDALocal(pu,&da_local));
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  PetscCall(DMDAGetElements(da_local,&nel,&nen,&elidx));

  for (np=0; np<npoints; np++) {
    PetscInt k,cidx,local_cidx;

    /* Cell index in parallel DMDA */
    cidx = cellid[np];
    /* Cell index in local DMDA */
    local_cidx = gtol_elidx[cidx];
    for (k=0; k<nen; k++) {
      PetscInt d,nidx;
      nidx = elidx[nen*local_cidx + k];

      PetscPrintf(
        PETSC_COMM_SELF,
        "Rank[%d]: Point[%" PetscInt_FMT "] Cell[%" PetscInt_FMT "] LocalCell[%" PetscInt_FMT "] Node[%" PetscInt_FMT "] Coords: (",
        rank,np,cidx,local_cidx,nidx
      );
      for (d=0; d<pu->dim; d++) {
        PetscPrintf(PETSC_COMM_SELF," %g ",(double)coords[pu->dim*np + d]);
      }
      PetscPrintf(PETSC_COMM_SELF,") Weight: %g\n",(double)weights[np*pu->npart + nidx] );
    }
    PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Point[%" PetscInt_FMT "] Fields: (",rank,np);
    for (k=0; k<bs; k++) {
      PetscPrintf(PETSC_COMM_SELF," %g ",fields[k*npoints + np]);
    }
    PetscPrintf(PETSC_COMM_SELF,")\n");
  }

  /* Restore fields */
  PetscCall(DMDARestoreElements(da_local,&nel,&nen,&elidx));
  PetscCall(DMSwarmRestoreField(swarm, coordFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(swarm, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swarm, "fields",&bs,NULL,(void **)&fields));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintDMDAInfo(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  PetscMPIInt      rank;
  DM               dap,das;
  PetscInt         dap_start[3],dap_size[3],das_start[3],das_size[3];
  PetscInt         dap_elstart[3],dap_elsize[3],das_elstart[3],das_elsize[3];
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[%s]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PUGetDMDALocal(pu,&das));

  PetscCall(DMDAGetCorners(dap,&dap_start[0],&dap_start[1],&dap_start[2],&dap_size[0],&dap_size[1],&dap_size[2]));
  PetscCall(DMDAGetCorners(das,&das_start[0],&das_start[1],&das_start[2],&das_size[0],&das_size[1],&das_size[2]));

  PetscCall(DMDAGetElementsCorners(dap,&dap_elstart[0],&dap_elstart[1],&dap_elstart[2]));
  PetscCall(DMDAGetElementsSizes(dap,&dap_elsize[0],&dap_elsize[1],&dap_elsize[2]));
  PetscCall(DMDAGetElementsCorners(das,&das_elstart[0],&das_elstart[1],&das_elstart[2]));
  PetscCall(DMDAGetElementsSizes(das,&das_elsize[0],&das_elsize[1],&das_elsize[2]));

  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Global DMDA: starts: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "), sizes: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")\n",
    rank,dap_start[0],dap_start[1],dap_start[2],dap_size[0],dap_size[1],dap_size[2]);
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Local  DMDA: starts: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "), sizes: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")\n",
    rank,das_start[0],das_start[1],das_start[2],das_size[0],das_size[1],das_size[2]);

  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Global DMDA Elements: starts: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "), sizes: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")\n",
    rank,dap_elstart[0],dap_elstart[1],dap_elstart[2],dap_elsize[0],dap_elsize[1],dap_elsize[2]);
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Local  DMDA Elements: starts: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "), sizes: (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")\n",
    rank,das_elstart[0],das_elstart[1],das_elstart[2],das_elsize[0],das_elsize[1],das_elsize[2]);


  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintPartitionMap(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               dap,das;
  PetscInt         npart,iel,nel_p,nen_p,nel_s,nen_s,k;
  PetscInt         *gtol_elidx;
  const PetscInt   *elidx_p,*elidx_s;
  PetscMPIInt      rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[%s]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  PetscCall(DMDAGetElements(dap,&nel_p,&nen_p,&elidx_p));
  PetscCall(DMDAGetElements(das,&nel_s,&nen_s,&elidx_s));
  
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  
  for (iel=0; iel<nel_p; iel++) {
    PetscInt local_cidx = gtol_elidx[iel];
    for (k=0; k<nen_p; k++) {
      PetscInt nidx = elidx_s[nen_s*local_cidx + k];
      PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Cell[%" PetscInt_FMT "], LocalCell[%" PetscInt_FMT "], nidx[%" PetscInt_FMT "]\n",
        rank,iel,local_cidx,nidx);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  Partition        *partitions;
  DM               swarm_data,swarm_interp;
  PetscInt         k,d,nsteps,step,global_npart,output_frequency=1;
  PetscInt         M[3];
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
  radius  = overlap / PetscPowReal(global_npart,1.0/pu->dim);//PetscSqrtReal(pu->npart);
  epsilon = 1.0 / radius;
  PetscPrintf(PETSC_COMM_WORLD,"[[ Box length: %g, overlap: %g, radius: %g, epsilon: %g ]]\n",
    box_length,overlap,radius,epsilon);
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  //PetscCall(PrintDMDAInfo(ctx));
  //PetscCall(PrintPartitionMap(ctx));
  PetscCall(CreateSwarmData(ctx, &swarm_data));
  PetscCall(SwarmOutput(ctx, swarm_data, 1, data_output, PETSC_FALSE));

  PetscCall(PartitionSetUp_All(partitions, pu->npart, swarm_data));

  PetscCall(CreateSwarmInterp(ctx, &swarm_interp));
  /* Interpolate */
  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));

  PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_FALSE));
  //PetscCall(DMSwarmPrint(ctx, swarm_interp));
#if 0
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
    PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
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
#endif
  
  /* Output parallel DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));
  PetscCall(PartitionOfUnityMeshView_Seq(pu));
  /* Free memory */
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_advection()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  Partition        *partitions;
  DM               swarm_data,swarm_interp;
  PetscInt         k,d,nsteps,step,global_npart,output_frequency=1;
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

  //PetscCall(Test());
  PetscCall(Test_advection());

  PetscCall(PetscFinalize());
  
  return 0;
}