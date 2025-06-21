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

PetscErrorCode SwarmOutputCoordsPetscVec(DM sw, const char *prefix)
{
  DMSwarmCellDM celldm;
  PetscViewer   viewer;
  Vec           coords;
  char          fname[PETSC_MAX_PATH_LEN];
  const char    *dm_name,**coordFields;
  PetscFunctionBeginUser;
  /* filename */
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s/%s_coords.pbvec",prefix,dm_name)); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_coords.pbvec", dm_name)); }
  
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, coordFields[0], &coords));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,fname));
  PetscCall(VecView(coords,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, coordFields[0], &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmOutputFieldsPetscVec(DM sw, const char *prefix)
{
  PetscViewer viewer;
  Vec         dvec;
  char        fname[PETSC_MAX_PATH_LEN];
  const char  *dm_name;
  PetscFunctionBeginUser;
  /* filename */
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s/%s_fields.pbvec",prefix,dm_name)); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_fields.pbvec", dm_name)); }
  /* Generate a global vector from fields */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "fields", &dvec));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,fname));
  PetscCall(VecView(dvec,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "fields", &dvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}