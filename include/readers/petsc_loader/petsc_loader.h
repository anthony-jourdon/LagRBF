#ifndef LagRBF_petsc_loader_h
#define LagRBF_petsc_loader_h

PetscErrorCode SwarmSetFieldFromPetscFile(DM sw, const char *fname, PetscInt *block, MPI_Comm comm);
PetscErrorCode SwarmSetFromPetscFiles(DM sw, const char **filenames, PetscInt nfiles, const char *coor_fname);
PetscErrorCode PetscLoadToSwarmFromFiles(LagRBFCtx ctx, DM sw);
PetscErrorCode PetscLoadDataToSwarmFromFiles(LagRBFCtx ctx, DM sw);

#endif