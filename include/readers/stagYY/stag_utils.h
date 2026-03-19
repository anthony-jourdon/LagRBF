#ifndef softname_stag_utils_h
#define softname_stag_utils_h

#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode StagGetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps);
PetscErrorCode StagRestoreTimeSteps(PetscInt **steps, PetscReal **time_steps);
PetscErrorCode StagGetFilelistFromOptions(PetscInt step, PetscInt *nfiles, char ***filenames);
PetscErrorCode StagRestoreFilelist(PetscInt nfiles, char ***filenames);
PetscErrorCode StagLoadDataToSwarmFromFiles(LagRBFCtx ctx, DM sw);

#endif