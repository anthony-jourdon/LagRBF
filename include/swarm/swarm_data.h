#ifndef __softname_swarm_data_h
#define __softname_swarm_data_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmDataCreate(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmDataAssignMigrateDuplicateGhostPoints(LagRBFCtx ctx, DM swarm);

#endif