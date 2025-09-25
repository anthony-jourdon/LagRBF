static const char help[] = "Test to read HDF5 files.\n\n";

#include <petsc.h>
#include "readers/reader_utils.h"
#include "readers/stagYY/hdf5_read_coor.h"
#include "readers/stagYY/hdf5_read_fields.h"
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm_data.h"
#include "swarm/swarm_output.h"

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

/*
  In parallel I don't know if can I do this better:
    Data (coordinates + fields) are loaded from HDF5 files.
    There is 1 file per StagYY mpi rank.
    I distribute these files among MY ranks (not stagYY) trying to balance the load of files between ranks.
    Nothing guaranties (and in general it is not guaranteed) that the points will end on the rank they have been loaded
    once SwarmMigrate() is called.
    Problem is that because I need to load fields every time steps, some points may not be on the rank they were originally loaded on.
    The only work around I found so far, is to re-size the swarm, reload the coords, load the fields and then call SwarmMigrate().
    But it means that this re-size is done at each time step. I don't know yet if it's a problem or not.
*/
static PetscErrorCode SwarmDataSetFields(DM swarm, StagFieldInfo cinfo, StagFieldInfo fdata)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
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
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  StagFieldInfo    field_data,cinfo;
  StagGridType     coor_type;
  DM               swarm_data;
  PetscInt         d,f,nfiles,nn,step,step_start,step_end,nsteps;
  PetscReal        time,dt;
  PetscReal        *time_entries;
  //PetscInt         npoints,ndims,nsteps,tstart;
  //PetscReal        box_length,overlap,radius,epsilon;
  PetscBool        found,exists;
  char             **fname_list,**coor_dset;
  char             **vfname_list; // velocity file names
  const char       *fields[] = {"fields"};
  PetscFunctionBeginUser;
  
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nfiles",&nfiles,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -nfiles not found. User must provide the number of files to read.");
  PetscCall(PetscMalloc1(nfiles, &fname_list)); 
  PetscCall(PetscMalloc1(nfiles, &vfname_list)); 
  PetscCall(PetscMalloc1(3, &coor_dset));
  for (f=0; f<nfiles; f++) { 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fname_list[f])); 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&vfname_list[f]));
  }
  for (d=0; d<3; d++) {
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&coor_dset[d]));
  }
  /* Coordinates */
  nn = nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_fname", fname_list, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -coor_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -coor_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",nfiles,nn);
  nn = 3;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_names", coor_dset, &nn, &found));
  /* Coordinate type, if nothing is given, the default is STAG_3D i.e., x,y,z normally ordered */
  coor_type = STAG_3D;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-coor_type",(PetscInt*)&coor_type,&found));

  /* Velocities */
  nn = nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-velo_fname", vfname_list, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -velo_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -velo_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",nfiles,nn);
  
  /* Steps */
  step_start = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_start",&step_start,&found));
  step_end = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_end",&step_end,&found));
  nsteps = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time_nsteps",&nsteps,&found));
  PetscMalloc1(nsteps,&time_entries);
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-time_entries",time_entries,&nsteps,&found));

  #if 0
  /* Backward loop time */
  for (step=step_end-1; step>=step_start; step--) {
    if (step-1 < step_start) { dt = 0.0; }
    else                     { dt = time_entries[step-1] - time_entries[step]; }
    PetscPrintf(PETSC_COMM_WORLD,"[Step %d] Time: %g, dt = %g\n",step,time_entries[step],dt);
  }

  PetscPrintf(PETSC_COMM_WORLD,"\n");
  /* Forward loop time */
  for (step=step_start; step<step_end; step++) {
    if (step == step_end-1) { dt = 0.0; }
    else                    { dt = time_entries[step+1] - time_entries[step]; }
    PetscPrintf(PETSC_COMM_WORLD,"[Step %d] Time: %g, dt = %g\n",step,time_entries[step],dt);
  }
#endif
#if 1

  /*
  Extract information from files and store into data structure
  Each rank reads a subset of files distributed among all ranks
  This is done only once for coordinates but other fields are only initialized, the data are read in the time loop
  */
  StagProcessCoordinates(PETSC_COMM_WORLD,(const char**)fname_list,nfiles,coor_dset,coor_type,&cinfo);
  HDF5_SetUpFieldData(PETSC_COMM_WORLD,(const char**)vfname_list,nfiles,&field_data);

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
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Create data points */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));

  for (step=step_start; step<step_end; step++) {
    time += step;
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFSetTime(ctx,time));

    HDF5_GetFieldData(PETSC_COMM_WORLD, (const char **)vfname_list, nfiles, step, field_data);

    PetscCall(SwarmDataSetFields(swarm_data, cinfo, field_data));
    /* Output data */
    PetscCall(SwarmOutput(ctx,swarm_data,1,fields,PETSC_FALSE));

  }

  


  for (f=0; f<nfiles; f++) { 
    PetscCall(PetscFree(fname_list[f])); 
    PetscCall(PetscFree(vfname_list[f]));
  }
  for (d=0; d<3; d++) { PetscCall(PetscFree(coor_dset[d])); }
  PetscCall(PetscFree(fname_list));
  PetscCall(PetscFree(vfname_list));
  PetscCall(PetscFree(coor_dset));
  PetscCall(PetscFree(time_entries));
  StagFieldInfoDestroy(&field_data);
  StagFieldInfoDestroy(&cinfo);
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
#endif
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