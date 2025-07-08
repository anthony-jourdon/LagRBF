#ifndef __softname_advection_ctx_h
#define __softname_advection_ctx_h

#include <petsc.h>

typedef enum {
  RK1=0, // First order Runge-Kutta
  RK2,   // Second order Runge-Kutta
  RK4,   // Fourth order Runge-Kutta
  DOPRI  // Dormand-Prince 5(4) method
} AdvectionScheme;

typedef struct _p_AdvectionCtx *AdvectionCtx;
typedef struct _p_LagRBFCtx *LagRBFCtx;

struct _p_AdvectionCtx {
  AdvectionScheme scheme; // advection scheme to use
  PetscReal       dt; // time step for advection
  PetscInt        order; // order of the advection scheme
  char            velocity_name[PETSC_MAX_PATH_LEN]; // name of the velocity field in the DMSwarm
  void            *user; // user-defined context for additional data
  PetscErrorCode (*destroy)(AdvectionCtx);
  PetscErrorCode (*advection)(LagRBFCtx,DM,DM);
};

PetscErrorCode LagRBF_AdvectionCtxCreate(LagRBFCtx ctx, AdvectionScheme scheme, const char *velocity_name);
PetscErrorCode LagRBF_AdvectionCtxDestroy(LagRBFCtx ctx);
/* Setters */
PetscErrorCode AdvectionSetTimeStep(AdvectionCtx ctx, PetscReal dt);
PetscErrorCode AdvectionSetVelocityName(AdvectionCtx ctx,const char *name);
/* Getters */
PetscErrorCode AdvectionGetTimeStep(AdvectionCtx ctx, PetscReal *dt);
PetscErrorCode AdvectionGetIntegrationOrder(AdvectionCtx ctx, PetscInt *order);
PetscErrorCode AdvectionGetVelocityName(AdvectionCtx ctx, const char **velocity_name);

#endif