#ifndef __softname_dormand_prince_4_h
#define __softname_dormand_prince_4_h

#include <petsc.h>

typedef struct {
  PetscReal dt_fac;
  PetscReal dt_fac_min;
  PetscReal dt_fac_max;
  PetscReal a_tol,r_tol;
  PetscReal error;
} DoPriCtx;

#endif