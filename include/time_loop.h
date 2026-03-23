#ifndef __softname_time_loop_h
#define __softname_time_loop_h

#include "lagrangian_rbf.h"

PetscErrorCode InternalForwardAdvectionLoop(LagRBFCtx ctx, DM swarm_data, DM swarm_interp, PetscReal time, PetscReal time_next);
PetscErrorCode ForwardTimeLoop(LagRBFCtx ctx, DM swarm_data, DM swarm_interp, PetscInt nsteps, PetscInt steps[], PetscReal time_steps[], PetscInt restart_idx);

#endif