#ifndef __softname_swarm_interp_h
#define __softname_swarm_interp_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw);

#endif