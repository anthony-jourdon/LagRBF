#include <petsc.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

PetscErrorCode CreateDirectory(const char dirname[])
{
  PetscMPIInt rank;
  int num,error_number;

  PetscFunctionBeginUser;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Generate a new directory on proc 0 */
  error_number = 0;
  if (rank == 0) {
    num = mkdir(dirname, S_IRWXU | S_IRWXG);
    error_number = errno;
  }
  PetscCall(MPI_Bcast(&num,1,MPI_INT,0,PETSC_COMM_WORLD));
  PetscCall(MPI_Bcast(&error_number,1,MPI_INT,0,PETSC_COMM_WORLD));

  if (error_number == EEXIST) {
    PetscPrintf(PETSC_COMM_WORLD,"Writing output to existing directory: %s \n",dirname);
  } else if (error_number == EACCES) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Write permission is denied for the parent directory in which the new directory is to be added");
  } else if (error_number == EMLINK) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The parent directory has too many links (entries)");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The file system doesn't have enough room to create the new directory");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The parent directory of the directory being created is on a read-only file system and cannot be modified");
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Created output directory: %s\n",dirname);
  }
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestDirectory(const char dirname[], char mode, PetscBool *_exists)
{
  PetscMPIInt rank;
  int i_exists = 0;
  PetscBool exists = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  if (rank == 0) {
    PetscCall(PetscTestDirectory(dirname,mode,&exists));
    i_exists = (int)exists;
  }
  PetscCall(MPI_Bcast(&i_exists,1,MPI_INT,0,PETSC_COMM_WORLD));

  if (i_exists == 1) { *_exists = PETSC_TRUE; }
  else               { *_exists = PETSC_FALSE; }
  PetscFunctionReturn(PETSC_SUCCESS);
}