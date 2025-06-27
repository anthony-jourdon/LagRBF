#ifndef __softname_swarm_interp_h
#define __softname_swarm_interp_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetCoordinatesFromOptions(PartitionOfUnity pu, DM sw);
PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx);
PetscErrorCode SwarmInterpAddMarkersFromOptions(LagRBFCtx ctx, DM sw);
#endif