#ifndef __softname_io_utils_h
#define __softname_io_utils_h

#include <petsc.h>

PetscErrorCode CreateDirectory(const char dirname[]);
PetscErrorCode TestDirectory(const char dirname[], char mode, PetscBool *_exists);
PetscErrorCode TestFile(const char filename[], char mode, PetscBool *_exists);
PetscErrorCode ParaviewPVDOpen(const char pvdfilename[]);
PetscErrorCode ParaviewPVDAppend(const char pvdfilename[], double time, const char datafile[], const char directory_name[]);
PetscErrorCode ParaviewPVDOpenAppend(PetscBool not_first, PetscInt step, const char pvdfilename[], double time, const char datafile[], const char directory_name[]);
PetscErrorCode VecLoadFromFile(Vec v, const char *name);
PetscErrorCode ISLoadFromFile(IS is, const char *name);
PetscErrorCode GetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps);
PetscErrorCode RestoreTimeSteps(PetscInt **steps, PetscReal **time_steps);
PetscErrorCode BuildFilenameFromTemplate(const char template_name[], const char directory[], const char prefix[], const char field[], const char separator[], PetscInt step, char out[]);
PetscErrorCode GetFilelistFromOptions(const char *prefix, PetscInt step, PetscInt *nfiles, char ***filenames, char **coorfname);
PetscErrorCode RestoreFilelist(PetscInt nfiles, char ***filenames, char **coorfname);
#endif