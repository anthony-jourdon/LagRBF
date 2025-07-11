static const char help[] = "Test to read HDF5 files.\n\n";

#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "io_utils.h"
#include "swarm_utils.h"
#include "swarm_data.h"
#include "readers/stag_hdf5.h"
/*
For petsc independant code, use the following includes:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
*/

#define DEBUG

struct _p_CoordinateData {
  double *coor;   // Coordinates
  int    n[3];    // Number of points in each dimension
  int    dim;     // Number of dimensions
  int    npoints; // Total number of points
};
typedef struct _p_CoordinateData *CoordinateData;

static PetscErrorCode CoordinateDataCreate_SingleFile(CoordinateData *coor_info)
{
  CoordinateData c;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&c));
  *coor_info = c;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CoordinateDataDestroy_SingleFile(CoordinateData *coor_info)
{
  CoordinateData c = *coor_info;
  PetscFunctionBeginUser;

  if (!c) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (c->coor) { PetscCall(PetscFree(c->coor)); }
  PetscCall(PetscFree(c));
  *coor_info = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CoordinateDataCreate(PetscInt nfiles, CoordinateData **coor_info)
{
  CoordinateData *c;
  PetscInt       f;
  PetscFunctionBeginUser;

  PetscCalloc1(nfiles, &c);
  for (f=0; f<nfiles; f++) { PetscCall(CoordinateDataCreate_SingleFile(&c[f])); }
  *coor_info = c;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CoordinateDataDestroy(PetscInt nfiles, CoordinateData **coor_info)
{
  CoordinateData *c = *coor_info;
  PetscInt       f;
  PetscFunctionBeginUser;
  if (!c) { PetscFunctionReturn(PETSC_SUCCESS); }

  for (f=0; f<nfiles; f++) { PetscCall(CoordinateDataDestroy_SingleFile(&c[f])); }
  PetscCall(PetscFree(c));
  *coor_info = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

void HDF5_GetCoordinateDimensions(const char *filename, CoordinateData coor_info)
{
  hid_t      plist_id,file_id,dset_id,space_id;
  hsize_t    *dims;
  int        ndims,d;
  const char *dset_name = "/X"; // Only one is enough to get dimensions


  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_SELF, MPI_INFO_NULL); // independent file access
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
  if (file_id < 0) {
    fprintf(stderr, "[[ ERROR ]] Failed to open file %s\n", filename);
    H5Pclose(plist_id);
    return;
  }
  dset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    fprintf(stderr, "[[ ERROR ]] In file %s, failed to open dataset %s\n", filename, dset_name);
    return;
  }
  space_id = H5Dget_space(dset_id);
  ndims    = H5Sget_simple_extent_ndims(space_id);
  dims     = malloc(ndims * sizeof(hsize_t));
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  /* Store dimensions in CoordinateData */
  coor_info->dim     = ndims;
  coor_info->npoints = 1;
  for (d=0; d<ndims; d++) {
    coor_info->npoints *= (int)dims[d];
    coor_info->n[d] = (int)dims[d]; 
  }
  /* Allocate memory for coordinates */
  coor_info->coor = malloc(coor_info->npoints * coor_info->dim * sizeof(double));
  // Free resources
  H5Sclose(space_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  H5Pclose(plist_id);
  free(dims);
  return;
}

void HDF5_GetCoordinatesFromFile(const char *filename, int rank, CoordinateData coor_info) {
  hid_t      plist_id,file_id;
  int        i;
  const char *datasets[] = {"/X", "/Y", "/Z"};

  /* Get coordinate dimensions */
  HDF5_GetCoordinateDimensions(filename, coor_info);

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  /* Independent file access */
  H5Pset_fapl_mpio(plist_id, MPI_COMM_SELF, MPI_INFO_NULL);
  /* Open the file */
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
  if (file_id < 0) {
    fprintf(stderr, "[Rank %d] Failed to open file %s\n", rank, filename);
    H5Pclose(plist_id);
    return;
  }

  for (i=0; i<coor_info->dim; i++) {
    hid_t      dset_id,space_id;
    hsize_t    *dims;
    herr_t     status;
    int        ndims;
    const char *dset_name = datasets[i];
    
    dset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
    if (dset_id < 0) {
      fprintf(stderr, "[Rank %d] Failed to open dataset %s\n", rank, dset_name);
      continue;
    }

    space_id = H5Dget_space(dset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    dims = malloc(ndims * sizeof(hsize_t));

    H5Sget_simple_extent_dims(space_id, dims, NULL);
    
    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &coor_info->coor[coor_info->npoints * i]);
    if (status < 0) { fprintf(stderr, "[Rank %d] Failed to read dataset %s\n", rank, dset_name); }
    
    H5Sclose(space_id);
    H5Dclose(dset_id);
  }
  H5Fclose(file_id);
  H5Pclose(plist_id);
}

static PetscErrorCode CoordinatesGetLocalSize(CoordinateData *cinfo, PetscInt nfiles, PetscInt *npoints, PetscInt *ndims)
{
  PetscInt f,np;
  int rank;
  PetscFunctionBeginUser;
  
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  np = 0;
  for (f=0; f<nfiles; f++) {
    CoordinateData c = cinfo[f];
    np += c->npoints;
#ifdef DEBUG
    PetscPrintf(PETSC_COMM_SELF,"[Rank %d][[ %s ]] File %d: npoints = %d, dim = (%d, %d, %d)\n", 
          rank, PETSC_FUNCTION_NAME, f, c->npoints, c->n[0], c->n[1], c->n[2]);
#endif
  }
  if (npoints) { *npoints = np; }
  if (ndims)   { *ndims   = cinfo[0]->dim; }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
Need to :
  - ?Not sure? Reshape from 3D to 2D coordinates when the model is 2D
  - Reshape from Fortran to C ordering for petsc
  - Merge coordinates from different files
*/
static PetscErrorCode ReshapeCoords(CoordinateData *cinfo, PetscInt nfiles, PetscReal *coor)
{
  PetscInt f,tnp;
  int rank;
  PetscFunctionBeginUser;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  tnp = 0;
  for (f=0; f<nfiles; f++) {
    CoordinateData c = cinfo[f];
    PetscInt lnp;

    PetscPrintf(PETSC_COMM_SELF,"[Rank %d] Reshaping file %d: npoints = %d, ndim = %d\n", 
          rank, f, c->npoints, c->dim);

    for (lnp=0; lnp<c->npoints; lnp++) {
      PetscInt d;
      for (d=0; d<c->dim; d++) {
        coor[c->dim*tnp + d] = c->coor[c->npoints*d + lnp];
      }
      tnp++;
    }
    PetscPrintf(PETSC_COMM_SELF,"[Rank %d] Reshaped file %d: tnp = %d\n", rank, f, tnp);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessFiles(const char *file_list[], PetscInt nfiles, CoordinateData **cinfo, PetscInt *nfiles_local)
{
  CoordinateData *coor_info;
  PetscInt files_per_rank,remaining_files,nlocal_files;
  PetscInt istart,iend,i,cnt;
  int rank,n_ranks;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &n_ranks));
  
  /* Distribute files among ranks */
  files_per_rank  = nfiles / n_ranks;
  remaining_files = nfiles % n_ranks;
  /* Each rank handles a contiguous block of files */
  if (rank < remaining_files) {
    istart = rank * (files_per_rank + 1);
    iend   = istart + (files_per_rank + 1);
  } else {
    istart = rank * files_per_rank + remaining_files;
    iend   = istart + files_per_rank;
  }
  nlocal_files = iend - istart;
  /* Create data structure to hold coordinates info */
  PetscCall(CoordinateDataCreate(nlocal_files, &coor_info));
  cnt = 0;
  for (i=istart; i<iend; i++) {
    int p;
    CoordinateData cinfo = coor_info[cnt];
#ifdef DEBUG
    printf("[Rank %d] Reading file: %s\n", rank, file_list[i]);
#endif

    HDF5_GetCoordinatesFromFile(file_list[i], rank, cinfo);

    #ifdef DEBUG
    printf("[Rank %d], File %d: npoints = %d, dim = (%d, %d, %d)\n", 
          rank, i, cinfo->npoints, cinfo->n[0], cinfo->n[1], cinfo->n[2]);
    //for (p=0; p<cinfo->npoints; p++) {
    //  printf("[Rank %d] Point %d = (%f, %f, %f)\n", rank, p, cinfo->coor[p], cinfo->coor[cinfo->npoints + p], cinfo->coor[2*cinfo->npoints + p]);
    //}
#endif
    cnt++;
  }
  *cinfo = coor_info;
  *nfiles_local = nlocal_files;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataPoints(LagRBFCtx ctx, CoordinateData *cinfo, PetscInt nfiles, DM *sw)
{
  DMSwarmCellDM celldm;
  DM            swarm;
  PetscInt      npoints,ndims;
  PetscReal     *coords;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  
  PetscCall(SwarmDataCreate(ctx,&swarm));
  /* Get the rank-local number of points */
  PetscCall(CoordinatesGetLocalSize(cinfo, nfiles, &npoints, &ndims));
  PetscCall(DMSwarmSetLocalSizes(swarm, npoints, -1));
  /* Set the coordinate fields */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access coordinates */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Set coordinates */
  PetscCall(ReshapeCoords(cinfo,nfiles,coords));
  /* Restore coordinates */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));

#if 0
  PetscInt p;
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  for (p=0; p<npoints; p++) {
    PetscInt d;
    PetscPrintf(PETSC_COMM_WORLD,"Point %d = (", p);
    for(d=0; d<ndims; d++) {
      PetscPrintf(PETSC_COMM_WORLD,"%f", coords[ndims*p + d]);
      if (d < ndims-1) PetscPrintf(PETSC_COMM_WORLD,", ");
    }
    PetscPrintf(PETSC_COMM_WORLD,")\n");
  }
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
#endif

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode Test0()
{
  CoordinateData *coor_info;
  PetscInt cnt,npoints,ndims, *n;
  PetscReal *coor;
  int  i,rank,n_ranks,nfiles,nlocal_files;  
  int files_per_rank,remaining_files,istart,iend;
  const char *path = "/mnt/c/Users/jourdon/Documents/Models/Stagyy/test_convection_mpi/hdf5";
  //const char *path = "/mnt/c/Users/jourdon/Documents/Models/Stagyy/test_yinyang_mpi/hdf5";
  const char *flist[] = {
    
    "00001_00064_00128_00001_NodeCoordinates.h5",
    "00001_00064_00128_00002_NodeCoordinates.h5",
    "00001_00064_00128_00003_NodeCoordinates.h5",
    "00001_00064_00128_00004_NodeCoordinates.h5"
    /*
    "00032_00096_00016_00001_NodeCoordinates.h5",
    "00032_00096_00016_00002_NodeCoordinates.h5",
    "00032_00096_00016_00003_NodeCoordinates.h5",
    "00032_00096_00016_00004_NodeCoordinates.h5"
    */
  };
  char file_list[4][4096]; // Adjust size as needed
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &n_ranks));
  nfiles = sizeof(flist) / sizeof(flist[0]);

  for (i=0; i<nfiles; i++) {
    snprintf(file_list[i], 4096, "%s/%s", path, flist[i]);
  }

  // Distribute files among ranks (round-robin or block-wise)
  files_per_rank  = nfiles / n_ranks;
  remaining_files = nfiles % n_ranks;
  // Each rank handles a contiguous block of files
  if (rank < remaining_files) {
    istart = rank * (files_per_rank + 1);
    iend   = istart + (files_per_rank + 1);
  } else {
    istart = rank * files_per_rank + remaining_files;
    iend   = istart + files_per_rank;
  }
  nlocal_files = iend - istart;
  /* Allocate memory to hold coordinate data */
  coor_info = malloc(nlocal_files * sizeof(CoordinateData));
  //PetscPrintf(PETSC_COMM_SELF, "[Rank %d] Handling files from %d to %d, (local: %d, total: %d files)\n", 
  //            rank, istart, iend, nlocal_files, nfiles);
  cnt = 0;
  for (i=istart; i<iend; i++) {
    int p;
    CoordinateData cinfo = &coor_info[cnt];

    memset(cinfo, 0, sizeof(CoordinateData)); // Initialize
    printf("[Rank %d] Reading file: %s\n", rank, file_list[i]);
    HDF5_GetCoordinatesFromFile(file_list[i], rank, cinfo);
    printf("[Rank %d], File %d: npoints = %d, dim = (%d, %d, %d)\n", 
          rank, i, cinfo->npoints, cinfo->n[0], cinfo->n[1], cinfo->n[2]);
    for (p=0; p<cinfo->npoints; p++) {
      printf("[Rank %d] Point %d = (%f, %f, %f)\n", rank, p, cinfo->coor[p], cinfo->coor[cinfo->npoints + p], cinfo->coor[2*cinfo->npoints + p]);
    }
    cnt++;
  }
  /* Count how much points there are on this rank */
  PetscCall(CoordinateGetNPoints(coor_info, nlocal_files, &npoints, &ndims));
  PetscPrintf(PETSC_COMM_SELF,"[Rank %d] Total points on this rank: %" PetscInt_FMT ", ndims: %" PetscInt_FMT "\n",rank, npoints, ndims);
  PetscMalloc1(npoints * ndims, &coor);
  // Reshape coordinates 
  PetscCall(ReshapeCoords(coor_info, nlocal_files, coor));
  /* Print some coordinates */
  for (i=0; i<npoints; i++) {
    PetscInt d;
    printf("[Rank %d] Point %d = (", rank, i);
    for (d=0; d<ndims; d++) {
      printf("%f", coor[ndims*i + d]);
      if (d < ndims-1) printf(", ");
    }
    printf(")\n");
  }
#if 0
  for (i=0; i<npoints; i++) {
    PetscInt d;
    printf("[Rank %d] Point %d = (", rank, i);
    for (d=0; d<ndims; d++) {
      printf("%f", coor[ndims*i + d]);
      if (d < ndims-1) printf(", ");
    }
    printf(")\n");
  }
#endif
  // Free allocated memory
  for (i=0; i<nlocal_files; i++) {
    CoordinateData *cinfo = &coor_info[i];
    free(cinfo->coor);
  }
  free(coor_info);
  
  


  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode Test(void)
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CoordinateData   *cinfo;
  DM               swarm_data;
  PetscInt         d,k,f,nfiles,nn,nfiles_local;
  PetscInt         npoints,ndims;
  PetscReal        box_length,overlap,radius,epsilon;
  PetscBool        found,exists;
  char             **fname_list;
  const char       *fields[] = {"fields"};
  PetscFunctionBeginUser;
  
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nfiles",&nfiles,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -nfiles not found. User must provide the number of files to read.");
  PetscCall(PetscMalloc1(nfiles, &fname_list));
  for (f=0; f<nfiles; f++) { PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fname_list[f])); }
  nn = nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_fname", fname_list, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -coor_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -coor_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",nfiles,nn);
  /* 
  Extract information from files and store into CoordinateData data structure 
  Each rank reads a subset of files distributed among all ranks
  */
  PetscCall(ProcessFiles((const char**)fname_list,nfiles,&cinfo,&nfiles_local));
  

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
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,cinfo,nfiles_local,&swarm_data));
  PetscCall(SwarmOutput(ctx,swarm_data,1,fields));

#if 0
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
#endif
  

  for (f=0; f<nfiles; f++) { PetscCall(PetscFree(fname_list[f])); }
  PetscCall(PetscFree(fname_list));
  PetscCall(CoordinateDataDestroy(nfiles_local, &cinfo));
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