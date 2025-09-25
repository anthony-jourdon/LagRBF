static const char help[] = "Test to interpolate and advect particles from StagYY output.\n\n";

#include <petsc.h>
#include "readers/reader_utils.h"
#include "readers/stagYY/hdf5_read_coor.h"
#include "readers/stagYY/hdf5_read_fields.h"
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm_data.h"
#include "swarm_interp.h"
#include "interpolation.h"
#include "partition_solve.h"
#include "advection/advection_ctx.h"
#include "swarm/swarm_output.h"

typedef struct {
  PetscInt nfiles;
  char **coor_fname;
  char **velo_fname;
  char **coor_names;
  StagGridType coor_type;
  PetscInt step_start,step_end,nsteps;
  PetscReal *time_entries;
} ModelData;

static PetscErrorCode ModelDataCreate(ModelData **mdata)
{
  ModelData *m;
  PetscFunctionBeginUser;
  PetscNew(&m);
  *mdata = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ModelDataDestroy(ModelData **mdata)
{
  ModelData *m = *mdata;
  PetscInt  i;
  PetscFunctionBeginUser;

  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  
  for (i=0; i<m->nfiles; i++) {
    if (m->coor_fname) PetscCall(PetscFree(m->coor_fname[i]));
    if (m->velo_fname) PetscCall(PetscFree(m->velo_fname[i]));
  }
  if (m->coor_fname) PetscCall(PetscFree(m->coor_fname));
  if (m->velo_fname) PetscCall(PetscFree(m->velo_fname));
  if (m->coor_names) {
    for (i=0; i<3; i++) { PetscCall(PetscFree(m->coor_names[i])); }
    PetscCall(PetscFree(m->coor_names));
  }
  if (m->time_entries) PetscCall(PetscFree(m->time_entries));
  PetscFree(m);
  *mdata = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadOptionsStagInput(ModelData *mdata)
{
  PetscInt  f,d,nn,type;
  PetscBool found;
  PetscFunctionBeginUser;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nfiles",&mdata->nfiles,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -nfiles not found. User must provide the number of files to read.");
  
  
  PetscCall(PetscMalloc1(mdata->nfiles,&mdata->coor_fname));
  PetscCall(PetscMalloc1(mdata->nfiles,&mdata->velo_fname)); 
  PetscCall(PetscMalloc1(3, &mdata->coor_names));
  for (f=0; f<mdata->nfiles; f++) { 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&mdata->coor_fname[f])); 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&mdata->velo_fname[f]));
  }
  for (d=0; d<3; d++) {
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&mdata->coor_names[d]));
  }
  /* Coordinates */
  nn = mdata->nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_fname", mdata->coor_fname, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -coor_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == mdata->nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -coor_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",mdata->nfiles,nn);
  nn = 3;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_names", mdata->coor_names, &nn, &found));
  /* Coordinate type, if nothing is given, the default is STAG_3D i.e., x,y,z normally ordered */
  mdata->coor_type = STAG_3D;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-coor_type",&type,&found));
  mdata->coor_type = (StagGridType)type;

  /* Velocities */
  nn = mdata->nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-velo_fname", mdata->velo_fname, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -velo_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == mdata->nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -velo_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",mdata->nfiles,nn);
  
  /* Steps */
  mdata->step_start = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_start",&mdata->step_start,&found));
  mdata->step_end = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_end",&mdata->step_end,&found));
  mdata->nsteps = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time_nsteps",&mdata->nsteps,&found));
  PetscMalloc1(mdata->nsteps,&mdata->time_entries);
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-time_entries",mdata->time_entries,&mdata->nsteps,&found));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterp_AddMarkersSphericalFromOptions(PartitionOfUnity pu, DM sw, InsertMode mode)
{
  DM         da;
  PetscInt   m[] = {0,0,1};
  PetscInt   d,dim;
  PetscReal  O[]={0.0, 0.0, 0.0};
  PetscReal  L[]={1.0, 1.0, 1.0};
  PetscBool  found;
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

  PetscCall(DMSwarmSetPointsSphericalCoordinates(sw,O,L,m,mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode SwarmDataSetCoords(DM swarm, StagFieldInfo cinfo)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;
  PetscReal     *coords;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(DMGetDimension(swarm,&dim));
  /* Set the rank-local number of points */
  PetscCall(DMSwarmSetLocalSizes(swarm, cinfo->npoints, -1));
  /* Set the coordinate field */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Set coordinates */
  PetscCall(StagReshapeCoordinates(dim, cinfo, coords));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagReshapeVelocity(PetscInt dim, StagFieldInfo fdata, PetscReal *velocity)
{
  PetscInt p,d;
  PetscFunctionBeginUser;
  for (p=0; p<fdata->npoints; p++) {
    for (d=0; d<dim; d++) {
      velocity[ dim*p + d ] = fdata->field[ fdata->bs*p + d ];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmDataSetFields(DM swarm, StagFieldInfo cinfo, StagFieldInfo fdata)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMGetDimension(swarm,&dim));
  /* Set the rank-local number of points */
  PetscCall(DMSwarmSetLocalSizes(swarm, cinfo->npoints, -1));
  /* Set the coordinate field */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "fields", NULL, NULL, (void **)&velocity));
  /* Set coordinates */
  PetscCall(StagReshapeCoordinates(dim, cinfo, coords));
  /* Set velocity */
  PetscCall(StagReshapeVelocity(dim,fdata,velocity));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&velocity));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  PartitionOfUnity pu;
  StagFieldInfo    cinfo,fdata;
  AdvectionCtx     adv_ctx;
  ModelData        *mdata;
  LagRBFCtx        ctx;
  Partition        *partitions;
  DM               swarm_data,swarm_interp;
  PetscInt         d,k,step,ninterp;
  PetscReal        dt,box_length,overlap,radius,epsilon;
  PetscBool        exists;
  const char       *output_fields[] = {"fields","pidx"};
  const char       *output_data[]   = {"fields"};
  PetscFunctionBeginUser;
  PetscCall(ModelDataCreate(&mdata));
  PetscCall(ReadOptionsStagInput(mdata));

  StagProcessCoordinates(PETSC_COMM_WORLD,(const char**)mdata->coor_fname,mdata->nfiles,mdata->coor_names,mdata->coor_type,&cinfo);
  HDF5_SetUpFieldData(PETSC_COMM_WORLD,(const char**)mdata->velo_fname,mdata->nfiles,&fdata);

  /* Create Partition mesh and data structures */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
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
  /* Create data points */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  /* Set swarm data coordinates */
  PetscCall(SwarmDataSetCoords(swarm_data,cinfo));
  
  /* Output DMDA */
  PetscCall(PartitionOfUnityMeshView(pu));

  /* Set all the partitions data structure */
  PetscPrintf(PETSC_COMM_WORLD,"[[ PartitionSetUp_All ]]\n");
  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));

  /* Create interpolation points */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));
  /* Set the first points, required to call at least once with INSERT_VALUES before ADD_VALUES */
  PetscCall(SwarmInterp_AddMarkersSphericalFromOptions(pu,swarm_interp,INSERT_VALUES));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  /* Set unique point index to the interpolation swarm */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&ninterp));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm_interp,0,ninterp));
  
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  /* Set time for the first time step */
  PetscCall(LagRBFSetTime(ctx,mdata->time_entries[mdata->step_start]));
  /* First dt, then Dormand-Price will compute it */
  dt = mdata->time_entries[mdata->step_start+1] - mdata->time_entries[mdata->step_start]; 
  dt *= 1e-3;
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  for (step=mdata->step_start; step<mdata->step_end; step++) {
    PetscReal time,next_time;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,mdata->time_entries[step]));
    
    if (step == mdata->step_end-1) {
      next_time = mdata->time_entries[step];
    } else { 
      next_time = mdata->time_entries[step+1];
    }

    HDF5_GetFieldData(PETSC_COMM_WORLD, (const char **)mdata->velo_fname, mdata->nfiles, step, fdata);
    PetscCall(SwarmDataSetFields(swarm_data, cinfo, fdata));
    
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFSetTime(ctx,mdata->time_entries[step]));
    /* Interpolate */
    PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    /* Output */
    PetscCall(SwarmOutput(ctx,swarm_interp,2,output_fields,PETSC_TRUE));
    PetscCall(SwarmOutput(ctx,swarm_data,1,output_data,PETSC_FALSE));
    /* Advect */
    PetscCall(LagRBFGetTime(ctx,&time));
    while (time < next_time) {
      /* Advect */
      adv_ctx->advection(ctx,swarm_interp,swarm_data);
      /* Update time */
      PetscCall(LagRBFGetTimeStep(ctx,&dt));
      time += dt;
      PetscCall(LagRBFSetTimeStep(ctx,dt));
    }
  }
  PetscCall(ModelDataDestroy(&mdata));
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