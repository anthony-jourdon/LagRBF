#ifndef __LagRBF_interface_adv_h
#define __LagRBF_interface_adv_h

#include "lagrangian_rbf.h"

LAGRBF_EXPORT PetscErrorCode lagrbf_advection_function(
  void *ctx,
  PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), 
  void *fn_ctx
);

LAGRBF_EXPORT PetscErrorCode lagrbf_advection_function_batch(
  void *ctx,
  PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), 
  void *fn_ctx
);

LAGRBF_EXPORT PetscErrorCode lagrbf_avect_analytical(
  void *ctx, 
  PetscReal time, 
  PetscReal time_next
);

LAGRBF_EXPORT PetscErrorCode lagrbf_advect_interpolate(
  void *ctx, 
  PetscReal time, 
  PetscReal time_next
);

#endif