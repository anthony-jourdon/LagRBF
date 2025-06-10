#ifndef __softname_io_utils_h
#define __softname_io_utils_h

#include <petsc.h>

PetscErrorCode CreateDirectory(const char dirname[]);
PetscErrorCode TestDirectory(const char dirname[], char mode, PetscBool *_exists);

#endif