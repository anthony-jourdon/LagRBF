#ifndef softname_stag_utils_h
#define softname_stag_utils_h

#include <petsc.h>

PetscErrorCode StagGetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps);
PetscErrorCode StagRestoreTimeSteps(PetscInt **steps, PetscReal **time_steps);
PetscErrorCode StagGetFilelistFromOptions(PetscInt step, PetscInt *nfiles, char ***filenames);
PetscErrorCode StagRestoreFilelist(PetscInt nfiles, char ***filenames);
#endif