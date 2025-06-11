#include <petsc.h>
#include "partition_of_unity.h"

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

static PetscErrorCode SwarmInterpCreate(PartitionOfUnity pu, DM *sw)
{
  DM       da,swarm;
  PetscInt dim;
  char     dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(SwarmCreatePIC(pu,&swarm));
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",pu->npart,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",pu->nfields,PETSC_REAL));
  // May need to add another set of fields that need to be advected but not interpolated
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static SwarmInterpSetCoordinates(PartitionOfUnity pu, DM sw)
{
  DM         da;
  PetscInt   *m;
  PetscInt   d;
  PetscReal  *O,*L; // O[3],L[3];
  PetscBool  found;
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetBoundingBox(da,O,L));
  PetscCall(DMGetOptionsPrefix(da,&prefix));
  d = pu->dim;
  PetscCall(PetscMalloc1(d,&m));
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-npi",m,&d,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snpi not found. User must provide the number of interpolation points in each dimension.",prefix);
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%snpi must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,INSERT_VALUES));

  PetscFree(m);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetUp(PartitionOfUnity pu, DM *sw)
{

}