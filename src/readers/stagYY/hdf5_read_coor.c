/* 
  Code in this file is PETSc independent on purpose so I can reuse it in other contexts if needed
*/

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "readers/stagYY/hdf5_read_fields.h"
#include "readers/stagYY/hdf5_read_coor.h"
#include "readers/reader_utils.h"

void StagProcessCoordinates(MPI_Comm comm, const char *file_list[], int nfiles, char **dataset_names, StagGridType grid_type, StagFieldInfo *coords_information)
{
  StagFieldInfo cinfo;
  DatasetList   datasets;
  int           total_npoints,offset;
  int           i,idx_start,idx_end,nlocal_files,npoints;
  int           *sub_idx;

  /* Create data structure to store coordinates */
  StagFieldInfoCreate(&cinfo);

  datasets = cinfo->datasets;
  datasets->count = 3; // always 3 coordinates even in 2D so we can use the same structure for 2D and 3D
  datasets->capacity = 3;
  datasets->names = (char**)malloc(datasets->count * sizeof(char*));
  for (i=0; i<datasets->count; i++) { datasets->names[i] = strdup(dataset_names[i]); }

  /* Determine how files are distributed among ranks */
  ReadersGetRankDistribution(comm,nfiles,&idx_start,&idx_end,&nlocal_files);

  /* We first loop over files to determine how much points the rank will process */
  total_npoints = 0;
  for (i=idx_start; i<idx_end; i++) {
    /* Read coordinate size */
    HDF5_GetFieldShape(file_list[i],datasets->names[0],NULL,&npoints);
    total_npoints += npoints;
  }
  /* Fill data structure */
  cinfo->npoints = total_npoints;
  cinfo->bs      = 3;
  cinfo->field   = (double*)malloc(total_npoints * cinfo->bs * sizeof(double));
  /* Substitution indices so vector quantities and coordinates from StagYY match */
  sub_idx = (int*)malloc(3 * sizeof(int));
  switch (grid_type) {
    case STAG_2DXZ:
      sub_idx[0] = 0;
      sub_idx[1] = 2;
      sub_idx[2] = 1;
      break;
    
    case STAG_2DYZ:
      sub_idx[0] = 1;
      sub_idx[1] = 2;
      sub_idx[2] = 0;
      break;

    case STAG_3D:
      sub_idx[0] = 0;
      sub_idx[1] = 1;
      sub_idx[2] = 2;
      break;

    default:
      sub_idx[0] = 0;
      sub_idx[1] = 1;
      sub_idx[2] = 2;
      break;
  }
  cinfo->user = (void*)sub_idx;
  cinfo->destroy = free;

  /* Loop over files and store coordinates */
  offset = 0;
  for (i=idx_start; i<idx_end; i++) {
    int d;
    /* Read coordinate dimensions */
    HDF5_GetFieldShape(file_list[i],datasets->names[0],NULL,&npoints);
    /* Directly read coordinates into the correct location in cinfo->field */
    for (d=0; d<cinfo->bs; d++) {
      HDF5_GetFieldFromFile(comm, file_list[i], datasets->names[d], &cinfo->field[d*cinfo->npoints + offset]);
    }
    offset += npoints;
  }
  *coords_information = cinfo;
  return;
}