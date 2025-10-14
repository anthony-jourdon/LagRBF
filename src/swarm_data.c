#include <petsc.h>
#include "lagrangian_rbf.h"
#include "swarm/swarm_utils.h"
#include "geodynamics_fields.h"

PetscErrorCode SwarmDataCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               swarm;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));

  // To Change -> need to use the local DMDA
  PetscCall(SwarmCreatePIC(pu,&swarm));
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"rbf",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}