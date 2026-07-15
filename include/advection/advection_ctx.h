#ifndef __LagRBF_advection_ctx_h
#define __LagRBF_advection_ctx_h

#include <petsc.h>
#include "lagrangian_rbf.h"

typedef enum {
  RK1=0, // First order Runge-Kutta
  RK2,   // Second order Runge-Kutta
  RK4,   // Fourth order Runge-Kutta
  DOPRI  // Dormand-Prince 5(4) method
} AdvectionScheme;

typedef enum {
  INTERPOLATION=0, // Velocity field obtained by interpolation
  ANALYTICAL       // Velocity field obtained from an analytical function
} VelocityType;

struct _p_AdvectionCtx {
  AdvectionScheme scheme; // advection scheme to use
  VelocityType    velocity_type; // method to obtain the velocity field
  PetscReal       dt; // time step for advection
  PetscInt        order; // order of the advection scheme
  char            velocity_name[PETSC_MAX_PATH_LEN]; // name of the velocity field in the DMSwarm
  void            *user; // user-defined context for additional data
  void            *user_fn; // user-defined context for the velocity function
  PetscErrorCode (*destroy)(AdvectionCtx);
  PetscErrorCode (*advection)(LagRBFCtx,DM,DM);
  PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*); // user-defined point-wise function to compute the velocity at given coordinates
  PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*); // user-defined batch function to compute the velocity at given coordinates
};

PetscErrorCode LagRBF_AdvectionCtxCreate(LagRBFCtx ctx, AdvectionScheme scheme, const char *velocity_name);
PetscErrorCode LagRBF_AdvectionCtxDestroy(LagRBFCtx ctx);
PetscErrorCode AdvectionEvaluateVelocity(LagRBFCtx ctx, DM swi, DM swd);
PetscErrorCode AdvectionCopyVelocityField(LagRBFCtx ctx, DM swarm);
PetscErrorCode AdvectionInitializeTimestepVelocity(LagRBFCtx ctx, DM swarm);
/* Setters */
PetscErrorCode AdvectionSetTimeStep(AdvectionCtx ctx, PetscReal dt);
PetscErrorCode AdvectionSetVelocityName(AdvectionCtx ctx,const char *name);
PetscErrorCode AdvectionSetVelocityFunction(AdvectionCtx ctx, PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), void *user);
PetscErrorCode AdvectionSetVelocityFunctionBatch(AdvectionCtx ctx, PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), void *user);
PetscErrorCode AdvectionSetVelocityType(AdvectionCtx ctx, VelocityType vtype);
/* Getters */
PetscErrorCode AvectionCtxGetScheme(AdvectionCtx ctx, AdvectionScheme *scheme);
PetscErrorCode AdvectionGetTimeStep(AdvectionCtx ctx, PetscReal *dt);
PetscErrorCode AdvectionGetIntegrationOrder(AdvectionCtx ctx, PetscInt *order);
PetscErrorCode AdvectionGetVelocityName(AdvectionCtx ctx, const char **velocity_name);
PetscErrorCode LagRBFGetAdvectionScheme(LagRBFCtx ctx, AdvectionScheme *scheme);
#endif