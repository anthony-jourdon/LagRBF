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
