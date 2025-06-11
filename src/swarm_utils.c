#include <petsc.h>
#include "partition_of_unity.h"
#include "swarm_utils.h"

PetscErrorCode SwarmCreatePIC(PartitionOfUnity pu, DM *sw)
{
  DM       da,swarm;
  PetscInt dim;
  PetscFunctionBeginUser;
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetDimension(da, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)da), &swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(swarm, da));
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
