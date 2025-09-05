#ifndef __softname_hdf5_read_coor_h
#define __softname_hdf5_read_coor_h

#include <hdf5.h>
#include "readers/stagYY/hdf5_read_fields.h"

/* Grid types, required to know how coords are stored */
typedef enum {
  STAG_2DXZ=0,
  STAG_2DYZ,
  STAG_3D
} StagGridType;

void StagProcessCoordinates(MPI_Comm comm, const char *file_list[], int nfiles, char **dataset_names, StagGridType grid_type, StagFieldInfo *coords_information);
#endif