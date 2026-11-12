/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    reader_utils.c

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#include <petsc.h>

void ReadersGetRankDistribution(MPI_Comm comm, int nfiles, int *idx_start, int *idx_end, int *nlocal_files)
{
  int files_per_rank,remaining_files;
  int istart,iend;
  int ierr,rank,n_ranks;
  
  ierr = MPI_Comm_rank(comm, &rank);
  if (ierr != MPI_SUCCESS) {
    fprintf(stderr, "[[ ERROR ]] Failed to get MPI rank\n");
    return;
  }
  ierr = MPI_Comm_size(comm, &n_ranks);
  if (ierr != MPI_SUCCESS) {
    fprintf(stderr, "[[ ERROR ]] Failed to get MPI size\n");
    return;
  }
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
  if (idx_start)    { *idx_start   = istart;        }
  if (idx_end)      { *idx_end     = iend;          }
  if (nlocal_files) { *nlocal_files = iend - istart; }
  return;
}
