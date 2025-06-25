#ifndef __softname_interpolation_h
#define __softname_interpolation_h

#include <petsc.h>
#include "lagrangian_rbf.h"

PetscErrorCode RBFInterpolation(LagRBFCtx ctx, DM swi, DM swd, const char *fieldname);

#endif