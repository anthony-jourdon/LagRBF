#ifndef __softname_swarm_interp_h
#define __softname_swarm_interp_h

#include "lagrangian_rbf.h"

PetscErrorCode SwarmInterpCreateAdvectionFields(LagRBFCtx ctx, DM swarm);
PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw);
//PetscErrorCode SwarmInterpSetWeights_Opt(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw);
PetscErrorCode SwarmInterpSetCoordinatesFromOptions(PartitionOfUnity pu, DM sw, InsertMode mode);
PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx);
PetscErrorCode SwarmInterpAddMarkersFromOptions(LagRBFCtx ctx, DM sw, InsertMode mode);
PetscErrorCode SwarmInterpLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmInterpLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm);
PetscErrorCode SwarmInterpLoadFromFile(LagRBFCtx ctx, DM sw, const char *dirname, PetscBool load_pidx);

#if defined (__AVX__)
PetscErrorCode SwarmInterpSetWeights_AVX256(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetWeights_AVX256_PointVectorized(LagRBFCtx ctx, DM sw);
#endif
#if defined (__AVX512F__)
PetscErrorCode SwarmInterpSetWeights_AVX512(LagRBFCtx ctx, DM sw);
PetscErrorCode SwarmInterpSetWeights_AVX512_PointVectorized(LagRBFCtx ctx, DM sw);
#endif

#endif