#ifndef LagRBF_stag_utils_h
#define LagRBF_stag_utils_h

#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode StagLoadDataToSwarmFromFiles(LagRBFCtx ctx, DM sw);
PetscErrorCode StagGetFilelistFromOptions(PetscInt step, PetscInt *nfiles, char ***filenames, PetscBool **is_scalar);
PetscErrorCode StagRestoreFilelist(PetscInt nfiles, char ***filenames, PetscBool **is_scalar);

#endif