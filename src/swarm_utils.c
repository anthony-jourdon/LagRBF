#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "swarm_utils.h"
#include "io_utils.h"

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

PetscErrorCode SwarmOutputCoordsPetscVec(DM sw, const char *directory, const char *prefix)
{
  DMSwarmCellDM celldm;
  PetscViewer   viewer;
  Vec           coords;
  char          fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char    *dm_name,**coordFields;
  PetscFunctionBeginUser;
  /* filename */
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_coords.pbvec",prefix,dm_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_coords.pbvec", dm_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }
  
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, coordFields[0], &coords));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(VecView(coords,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, coordFields[0], &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmOutputFieldsPetscVec(DM sw, const char *directory, const char *prefix)
{
  PetscViewer viewer;
  Vec         dvec;
  char        fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char  *dm_name;
  PetscFunctionBeginUser;
  /* filename */
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_fields.pbvec",prefix,dm_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_fields.pbvec", dm_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }
  /* Generate a global vector from fields */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "fields", &dvec));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(VecView(dvec,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "fields", &dvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw)
{
  PetscBool        exists;
  static PetscBool beenhere=PETSC_FALSE;
  char             root[PETSC_MAX_PATH_LEN],outputdir[PETSC_MAX_PATH_LEN],pvdfname[PETSC_MAX_PATH_LEN],vtkfname[PETSC_MAX_PATH_LEN];
  char             stepprefix[PETSC_MAX_PATH_LEN],stepdir[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  
  PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",ctx->output_dir);
  PetscSNPrintf(stepdir,PETSC_MAX_PATH_LEN-1,"step%" PetscInt_FMT "",ctx->step);
  PetscSNPrintf(outputdir,PETSC_MAX_PATH_LEN-1,"%s/%s",root,stepdir);
  /* Check if directory exists */
  PetscCall(TestDirectory(outputdir,'w',&exists));
  /* If not create it */
  if (!exists) { PetscCall(CreateDirectory(outputdir)); }
  /* PVD */
  PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%06" PetscInt_FMT "",ctx->step);
  PetscSNPrintf(pvdfname,PETSC_MAX_PATH_LEN-1,"%s/timeseries_X.pvd",root);
  PetscSNPrintf(vtkfname, PETSC_MAX_PATH_LEN-1, "%s_coords.pbvec",stepprefix);

  if (!beenhere) { PetscPrintf(PETSC_COMM_WORLD,"  writing pvdfilename %s \n", pvdfname ); }
  PetscCall(ParaviewPVDOpenAppend(beenhere,ctx->step,pvdfname,ctx->time, vtkfname, stepdir));
  beenhere = PETSC_TRUE;

  PetscCall(SwarmOutputCoordsPetscVec(sw,outputdir,stepprefix));
  PetscCall(SwarmOutputFieldsPetscVec(sw,outputdir,stepprefix));

  PetscFunctionReturn(PETSC_SUCCESS);
}