#ifndef __softname_reader_utils_h
#define __softname_reader_utils_h

#include <petsc.h>
#include "readers/stagYY/hdf5_read_fields.h"
#include "readers/stagYY/stag_utils.h"

void ReadersGetRankDistribution(MPI_Comm comm, int nfiles, int *idx_start, int *idx_end, int *nlocal_files);
PetscErrorCode StagReshapeCoordinates(PetscInt dim, StagFieldInfo cinfo, PetscReal *coor);

#endif