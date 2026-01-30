#include <petsc.h>
#include <petsc/private/dmswarmimpl.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "swarm/swarm_output.h"
#include "io_utils.h"

static PetscErrorCode SwarmOutput_VecCreate(DM sw, const char *field_name, PetscBool permute, Vec *vec)
{
  Vec dvec;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, field_name, &dvec));
  if (permute) {
    Vec       new_vec;
    PetscInt  npoints,bs,np,k;
    PetscReal *LA_values,*_values;
    /* 
      Create a new vector to hold the permuted values
      We need this because DMSwarmCreateGlobalVectorFromField shares the 
      array with the field, but permutation is only for output.
    */
    PetscCall(VecDuplicate(dvec,&new_vec));
    PetscCall(VecCopy(dvec,new_vec));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, field_name, &dvec));
    /* Permute entries in the vector from Fortran-like ordering to C-like ordering */
    PetscCall(DMSwarmGetLocalSize(sw,&npoints));
    PetscCall(VecGetArray(new_vec,&LA_values));
    PetscCall(DMSwarmGetField(sw,field_name,&bs,NULL,(void **)&_values));
    for (np=0; np<npoints; np++) {
      PetscInt Fidx,Cidx;
      for (k=0; k<bs; k++) {
        Fidx = k*npoints + np;
        Cidx = np*bs + k;
        LA_values[Cidx] = _values[Fidx];
      }
    }
    PetscCall(DMSwarmRestoreField(sw,field_name,&bs,NULL,(void **)&_values));
    PetscCall(VecRestoreArray(new_vec,&LA_values));
    *vec = new_vec;
  } else {
    *vec = dvec;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmOutput_VecDestroy(DM sw, const char *field_name, PetscBool permute, Vec *vec)
{
  Vec v = *vec;
  PetscFunctionBeginUser;
  if (permute) {
    PetscCall(VecDestroy(&v));
  } else {
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, field_name, &v));
  }
  *vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmOutputFieldPetscIS_FromName(DM sw, const char *field_name, const char *directory, const char *prefix)
{
  PetscViewer    viewer;
  IS             is;
  PetscInt       npoints;
  PetscInt       *_idx = NULL;
  const PetscInt *idx;
  PetscDataType  type;
  MPI_Comm       comm;
  void           *data;
  char           fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char     *dm_name;
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetComm((PetscObject)sw,&comm));

  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_%s.pbis",prefix,dm_name,field_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_%s.pbis",dm_name,field_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetField(sw,field_name,NULL,&type,&data));
  if (sizeof(PetscInt) == sizeof(PetscInt64)) {
    idx = (const PetscInt *)data;
  } else if (type == PETSC_INT64) {
    PetscInt64 *data64 = (PetscInt64 *)data;
    PetscInt   np;
    PetscMalloc1(npoints,&_idx);
    for (np=0; np<npoints; np++) {
      PetscIntCast(data64[np],&_idx[np]);
    }
    idx = (const PetscInt *)_idx;
  } else {
    idx = (const PetscInt *)data;
  }
  PetscCall(ISCreateGeneral(comm,npoints,idx,PETSC_USE_POINTER,&is));
  PetscCall(PetscObjectSetName((PetscObject)is, field_name));
  /* Output */
  PetscCall(PetscViewerCreate(comm,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(ISView(is,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(ISDestroy(&is));
  PetscCall(DMSwarmRestoreField(sw,field_name,NULL,&type,&data));
  if (_idx) PetscFree(_idx);
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
TODO: A special case for fields that are not of type PETSC_REAL is required.
*/
PetscErrorCode SwarmOutputFieldPetscVec_FromName(DM sw, const char *field_name, const char *directory, const char *prefix, PetscBool permute)
{
  PetscViewer viewer;
  Vec         new_vec=NULL;
  char        fname[PETSC_MAX_PATH_LEN],outputname[PETSC_MAX_PATH_LEN];
  const char *dm_name;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  if (prefix) { PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s.%s_%s.pbvec",prefix,dm_name,field_name); }
  else        { PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1,"%s_%s.pbvec",dm_name,field_name)); }
  /* If directory is provided, prepend it to the filename */
  if (directory) { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s/%s",directory,fname); }
  else           { PetscSNPrintf(outputname, PETSC_MAX_PATH_LEN-1,"%s",fname); }

  PetscCall(SwarmOutput_VecCreate(sw,field_name,permute,&new_vec));
  /* Output */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)sw),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,outputname));
  PetscCall(VecView(new_vec,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* Clean-up */
  PetscCall(SwarmOutput_VecDestroy(sw,field_name,permute,&new_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmOutput(LagRBFCtx ctx, DM sw, PetscInt nfields, const char *fields[], PetscBool permute)
{
  PetscInt         f;
  PetscBool        exists;
  static PetscBool beenhere=PETSC_FALSE;
  char             root[PETSC_MAX_PATH_LEN],outputdir[PETSC_MAX_PATH_LEN],pvdfname[PETSC_MAX_PATH_LEN],vtkfname[PETSC_MAX_PATH_LEN];
  char             stepprefix[PETSC_MAX_PATH_LEN],stepdir[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  
  PetscCall(PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",ctx->output_dir));
  PetscCall(PetscSNPrintf(stepdir,PETSC_MAX_PATH_LEN-1,"step%" PetscInt_FMT "",ctx->step));
  PetscCall(PetscSNPrintf(outputdir,PETSC_MAX_PATH_LEN-1,"%s/%s",root,stepdir));
  /* Check if directory exists */
  PetscCall(TestDirectory(outputdir,'w',&exists));
  /* If not create it */
  if (!exists) { PetscCall(CreateDirectory(outputdir)); }
  /* PVD */
  PetscCall(PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%06" PetscInt_FMT "",ctx->step));
  PetscCall(PetscSNPrintf(pvdfname,PETSC_MAX_PATH_LEN-1,"%s/timeseries_X.pvd",root));
  PetscCall(PetscSNPrintf(vtkfname, PETSC_MAX_PATH_LEN-1, "%s_coords.pbvec",stepprefix));

  if (!beenhere) { PetscPrintf(PETSC_COMM_WORLD,"  writing pvdfilename %s \n", pvdfname ); }
  PetscCall(ParaviewPVDOpenAppend(beenhere,ctx->step,pvdfname,ctx->time, vtkfname, stepdir));
  beenhere = PETSC_TRUE;

  if (ctx->forward)  { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_forward", ctx->step)); }
  if (ctx->backward) { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_backward", ctx->step)); }

  /* If Swarm type is PIC: */
  { /* Coordinates */
    DM_Swarm *swarm = (DM_Swarm*)sw->data;
    if (swarm->swarm_type == DMSWARM_PIC) {
      DMSwarmCellDM celldm;
      const char    **coordFields;
      PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
      PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
      PetscCall(SwarmOutputFieldPetscVec_FromName(sw,coordFields[0],outputdir,stepprefix,PETSC_FALSE));
    }
  }
  for (f=0; f<nfields; f++) {
    PetscDataType type;
    void          *data;
    PetscCall(DMSwarmGetField(sw,fields[f],NULL,&type,&data));
    PetscCall(DMSwarmRestoreField(sw,fields[f],NULL,&type,&data));
    switch (type) {
    case PETSC_REAL:
      PetscCall(SwarmOutputFieldPetscVec_FromName(sw,fields[f],outputdir,stepprefix,permute));
      break;
    
    case PETSC_INT:
    case PETSC_INT64:
      PetscCall(SwarmOutputFieldPetscIS_FromName(sw,fields[f],outputdir,stepprefix));
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_UNKNOWN_TYPE,"Data type must be PETSC_REAL or PETSC_INT");
      break;
    }    
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}