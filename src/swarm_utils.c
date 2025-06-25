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

PetscErrorCode SwarmOutputFieldPetscIS_FromName(DM sw, const char *field_name, const char *directory, const char *prefix)
{
  PetscViewer    viewer;
  IS             is;
  PetscInt       npoints;
  const PetscInt *idx;
  PetscDataType  type;
  void           *data;
  char           fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char     *dm_name;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_%s.pbis",prefix,dm_name,field_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_%s.pbis",dm_name,field_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetField(sw,field_name,NULL,&type,&data));
  idx = (const PetscInt *)data;

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)sw), npoints, idx, PETSC_USE_POINTER, &is));
  PetscCall(PetscObjectSetName((PetscObject)is, field_name));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(ISView(is,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(ISDestroy(&is));
  PetscCall(DMSwarmRestoreField(sw,field_name,NULL,&type,&data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
TODO: A special case for fields that are not of type PETSC_REAL is required.
*/
PetscErrorCode SwarmOutputFieldPetscVec_FromName(DM sw, const char *field_name, const char *directory, const char *prefix)
{
  PetscViewer viewer;
  Vec         dvec;
  char        fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char *dm_name;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_%s.pbvec",prefix,dm_name,field_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_%s.pbvec",dm_name,field_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }

  /* Generate a global vector from fields */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, field_name, &dvec));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(VecView(dvec,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, field_name, &dvec));
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

PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw, PetscInt nfields, const char *fields[])
{
  PetscInt         f;
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
  //PetscCall(SwarmOutputFieldsPetscVec(sw,outputdir,stepprefix));
  for (f=0; f<nfields; f++) {
    PetscDataType type;
    void          *data;
    PetscCall(DMSwarmGetField(sw,fields[f],NULL,&type,&data));
    PetscCall(DMSwarmRestoreField(sw,fields[f],NULL,&type,&data));
    switch (type) {
    case PETSC_REAL:
      PetscCall(SwarmOutputFieldPetscVec_FromName(sw,fields[f],outputdir,stepprefix));
      break;
    
    case PETSC_INT:
      PetscCall(SwarmOutputFieldPetscIS_FromName(sw,fields[f],outputdir,stepprefix));
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_UNKNOWN_TYPE,"Data type must be PETSC_REAL or PETSC_INT");
      break;
    }    
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}