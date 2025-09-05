#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include "readers/reader_utils.h"
#include "readers/stagYY/hdf5_read_fields.h"

void DataSetListCreate(DatasetList *dset)
{
  DatasetList d;

  /* Allocate data structure */
  d = (DatasetList)malloc(sizeof(struct _p_DatasetList));
  /* Clear memory */
  memset(d, 0, sizeof(struct _p_DatasetList));
  *dset = d;
  
  return;
}

void DataSetListDestroy(DatasetList *dset)
{
  DatasetList d = *dset;
  int         i;

  if (!d) return;
  /* Free dataset names */
  for (i=0; i<d->count; i++) { free(d->names[i]); }
  free(d->names);
  /* Free struct */
  free(d);
  *dset = NULL;

  return;
}

void StagFieldInfoCreate(StagFieldInfo *field_data)
{
  StagFieldInfo f;
  
  /* Allocate field_data struct */
  f = (StagFieldInfo)malloc(sizeof(struct _p_StagFieldInfo));
  /* Clear memory */
  memset(f, 0, sizeof(struct _p_StagFieldInfo));
  /* Create dataset list */
  DataSetListCreate(&f->datasets);
  *field_data = f;

  return;
}

void StagFieldInfoDestroy(StagFieldInfo *field_data)
{
  StagFieldInfo f = *field_data;

  if (!f) return;
  /* Free pointers here before freeing the struct */
  if (f->field)    free(f->field);
  if (f->datasets) DataSetListDestroy(&f->datasets);
  /* Call user-defined destroy function if it exists */
  if (f->destroy && f->user) f->destroy(f->user);
  /* Free struct */
  free(f);
  *field_data = NULL;

  return;
}

/* Define a callback function used to extract time step datasets */
static herr_t collect_tstep_datasets(hid_t loc_id, const char *name, const H5L_info_t *info, void *op_data) {
  DatasetList list = (DatasetList)op_data;
  H5O_info_t  obj_info;
  regex_t     regex;

  // Make sure it is a dataset
  if (H5Oget_info_by_name(loc_id, name, &obj_info, H5P_DEFAULT) < 0) { return 0; }
  if (obj_info.type != H5O_TYPE_DATASET) { return 0; }

  // Match any name ending with an underscore and 5 digits (e.g., *_00001)
  // TODO: check if we may need to set the matching char in the op_data struct to generalize this regex research
  regcomp(&regex, ".*_[0-9]{5}$", REG_EXTENDED);
  int match = regexec(&regex, name, 0, NULL, 0);
  regfree(&regex);
  if (match != 0) { return 0; }

  // Grow array
  if (list->count == list->capacity) {
    size_t newcap = list->capacity ? list->capacity * 2 : 16;
    char **newnames = realloc(list->names, newcap * sizeof(char *));
    if (!newnames) {
      fprintf(stderr, "Memory allocation failed\n");
      return -1; // Stop iteration on error
    }
    list->names = newnames;
    list->capacity = newcap;
  }

  list->names[list->count] = strdup(name);
  if (!list->names[list->count]) {
    fprintf(stderr, "strdup failed\n");
    return -1;
  }
  list->count++;
  return 0; // continue iteration
}

void HDF5_GetFieldDatasets(MPI_Comm comm, const char *filename, DatasetList dset_list)
{
  hid_t  plist_id,file_id;
  herr_t status;
  int    ierr,rank;

  ierr = MPI_Comm_rank(comm, &rank);
  if (ierr != MPI_SUCCESS) {
    fprintf(stderr, "[[ ERROR ]] Failed to get MPI rank\n");
    return;
  }

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  /* Independent file access */
  H5Pset_fapl_mpio(plist_id, MPI_COMM_SELF, MPI_INFO_NULL);
  /* Open the file */
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
  /* Discover all datasets ending in "_%05d" (5 digits) */
  status = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_INC, NULL, collect_tstep_datasets, (void*)dset_list);
  if (status < 0) { fprintf(stderr, "[Rank %d] Failed to read dataset from file %s\n", rank, filename); }

  H5Pclose(plist_id);
  H5Fclose(file_id);
}

void HDF5_GetFieldShape(const char *filename, const char *dset_name, int *bs, int *npoints)
{
  hid_t   plist_id,file_id,dset_id,space_id;
  hsize_t shape[10]; // if it happens that blocksize > 10 a dynamic allocation should be prefered, but for stag it's unlikely
  int     blocksize;

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
  space_id  = H5Dget_space(dset_id);
  blocksize = H5Sget_simple_extent_dims(space_id, shape, NULL);

  /* Very StagYY specific, may not be re-usable for any HDF5 dataset */
  switch (blocksize) {
  case 2: // scalar 2D
    if (bs)      *bs = 1;
    if (npoints) *npoints = shape[0] * shape[1];
    break;

  case 3: // scalar 3D
    if (bs)      *bs = 1;
    if (npoints) *npoints = shape[0] * shape[1] * shape[2];
    break;

  case 4: // vector 2D and 3D
    if (bs)      *bs = shape[3];
    if (npoints) *npoints = shape[0] * shape[1] * shape[2];
    break;

  default:
    fprintf(stderr, "[[ ERROR ]] In file %s, unsupported blocksize %d for dataset %s\n", filename, blocksize, dset_name);
    break;
  }

  H5Sclose(space_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  H5Pclose(plist_id);
  return;
}

void HDF5_GetFieldFromFile(MPI_Comm comm, const char *filename, const char *dset_name, double *field) 
{
  hid_t      plist_id,file_id,dset_id;
  herr_t     status;
  int        ierr,rank;

  ierr = MPI_Comm_rank(comm, &rank);
  if (ierr != MPI_SUCCESS) {
    fprintf(stderr, "[[ ERROR ]] Failed to get MPI rank\n");
    return;
  }

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
  /* Open the dataset */
  dset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    fprintf(stderr, "[Rank %d] Failed to open dataset %s\n", rank, dset_name);
    return;
  }
  /* Get coordinates */
  status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
  if (status < 0) { fprintf(stderr, "[Rank %d] Failed to read dataset %s\n", rank, dset_name); }
  /* Close resources */
  H5Dclose(dset_id);
  H5Fclose(file_id);
  H5Pclose(plist_id);
}

void HDF5_SetUpFieldData(MPI_Comm comm, const char *file_list[], int nfiles, StagFieldInfo *field_data)
{
  StagFieldInfo fdata;
  DatasetList   datasets;
  int           i,idx_start,idx_end,nlocal_files;
  int           bs,npoints,total_npoints,dset_offset;

  /* Create the data structure corresponding to this field */
  StagFieldInfoCreate(&fdata);
  datasets = fdata->datasets;
  /* Determine how files are distributed among ranks */
  ReadersGetRankDistribution(comm,nfiles,&idx_start,&idx_end,&nlocal_files);
  /* Loop over files and collect dataset names */
  total_npoints = 0;
  dset_offset   = 0;
  for (i=idx_start; i<idx_end; i++) {
    HDF5_GetFieldDatasets(comm,file_list[i],datasets);
    printf("datasets->names[%d] = %s\n", dset_offset, datasets->names[dset_offset]);
    HDF5_GetFieldShape(file_list[i], datasets->names[dset_offset], &bs, &npoints);
    total_npoints += npoints;
    dset_offset = datasets->count;
  }
  /* Fill data structure */
  fdata->field   = (double*)malloc(bs * total_npoints * sizeof(double));
  fdata->bs      = bs;
  fdata->npoints = total_npoints;

  *field_data = fdata;
  return;
}

void HDF5_GetFieldData(MPI_Comm comm, const char *file_list[], int nfiles, int step, StagFieldInfo field_data)
{
  DatasetList datasets;
  int i,f,idx_start,idx_end,nlocal_files,dset_per_file;

  datasets = field_data->datasets;
  /* Determine how files are distributed among ranks */
  ReadersGetRankDistribution(comm,nfiles,&idx_start,&idx_end,&nlocal_files);
  dset_per_file = datasets->count / nlocal_files;

  f = 0;
  for (i=idx_start; i<idx_end; i++) {
    int        bs,npoints;
    const char *dset_name = datasets->names[f * dset_per_file + step];

    HDF5_GetFieldShape(file_list[i], dset_name, &bs, &npoints);
    HDF5_GetFieldFromFile(comm, file_list[i], dset_name, &field_data->field[f * bs * npoints]);
    f++;
  }
  return;
}