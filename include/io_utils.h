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
#endif