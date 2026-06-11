#include "lagrangian_rbf.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_utils.h"
#include "readers/stagYY/binary_loader.h"
#include "readers/stagYY/stag_utils.h"
#include "cell_info.h"
#include "io_utils.h"

PetscErrorCode StagGetFilelistFromOptions(PetscInt step, PetscInt *nfiles, char ***filenames, PetscBool **is_scalar)
{
  PetscInt    nn,f;
  char        **fieldnames = NULL;
  char        prefix[PETSC_MAX_PATH_LEN];
  PetscBool   found,found_vp;
  PetscBool   *scalar;
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"data_"));
  /* Assemble the list of files to read */
  PetscCall(GetFilelistFromOptions((const char *)prefix,step,nfiles,filenames,NULL));
  /* Array to know which fields are scalar and which are vector */
  PetscCall(PetscMalloc1(*nfiles,&scalar));
  PetscCall(PetscMalloc1(*nfiles,&fieldnames));
  nn = *nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-data_fname_fields", fieldnames, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -data_fname_fields not found. User must provide the list of field names to read.");
  PetscCheck(nn == *nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -data_fname_fields must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",*nfiles,nn);
  
  for (f=0; f<*nfiles; f++) {
    PetscCall(PetscStrcmp(fieldnames[f],"vp",&found_vp));
    if (found_vp) {
      scalar[f] = PETSC_FALSE;
    } else {
      scalar[f] = PETSC_TRUE;
    }
  }
  if (is_scalar) *is_scalar = scalar; 
  PetscCall(PetscFree(fieldnames));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagRestoreFilelist(PetscInt nfiles, char ***filenames, PetscBool **is_scalar)
{
  PetscBool *scalar = *is_scalar;
  PetscFunctionBeginUser;
  PetscCall(RestoreFilelist(nfiles,filenames,NULL));
  if (scalar) {
    PetscCall(PetscFree(scalar));
    *is_scalar = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagLoadDataToSwarmFromFiles(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  PetscMPIInt      rank;
  PetscInt         nfiles,step;
  PetscBool        *is_scalar;
  char             **filenames;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  PetscCall(LagRBFGetStep(ctx,&step));

  PetscCall(StagGetFilelistFromOptions(step,&nfiles,&filenames,&is_scalar));
  /* Read data only on rank 0 */
  if (rank == 0) {
    PetscCall(StagBinSwarmSetFromFiles(sw,filenames,nfiles,is_scalar));
  }
  /* Ensure that rank 0 has finished reading data */
  PetscCall(PetscBarrier((PetscObject)NULL));
  /* Free filenames */
  PetscCall(StagRestoreFilelist(nfiles,&filenames,&is_scalar));
  /* Send data to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx,sw));
  /* Duplicate ghost points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,sw));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  /* Set up CellInfo data structure */
  PetscCall(CellInfoSetUp(pu,ci,sw));

  PetscFunctionReturn(PETSC_SUCCESS);
}