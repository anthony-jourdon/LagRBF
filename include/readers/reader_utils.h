#ifndef __LagRBF_reader_utils_h
#define __LagRBF_reader_utils_h

#include <petsc.h>
#include "readers/stagYY/stag_utils.h"

void ReadersGetRankDistribution(MPI_Comm comm, int nfiles, int *idx_start, int *idx_end, int *nlocal_files);

#endif