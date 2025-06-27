#include <petsc.h>
#include <petsc/private/dmswarmimpl.h>
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

static PetscErrorCode PointSetSphericalToCartesianCoordinate(PetscReal params[], PetscReal coords[])
{
  PetscFunctionBeginUser;
  coords[0] = params[0] * PetscSinReal(params[1]) * PetscCosReal(params[2]);
  coords[1] = params[0] * PetscSinReal(params[1]) * PetscSinReal(params[2]);
  coords[2] = params[0] * PetscCosReal(params[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointSetPolarToCartesianCoordinate(PetscReal params[], PetscReal coords[])
{
  PetscFunctionBeginUser;
  coords[0] = params[0] * PetscCosReal(params[1]);
  coords[1] = params[0] * PetscSinReal(params[1]);
  coords[2] = 0.0; /* z-coordinate is zero in polar coordinates */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetPointsSphericalCoordinates - Set point coordinates in a `DMSWARM` on a regular (ijk) spherical grid

  Collective

  Input Parameters:
+ sw      - the `DMSWARM`
. min     - minimum coordinate values in the r, theta, phi directions (array of length dim)
. max     - maximum coordinate values in the r, theta, phi directions (array of length dim)
. npoints - number of points in each spatial direction (array of length dim)
- mode    - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)

  Notes:
  When using mode = `INSERT_VALUES`, this method will reset the number of particles in the `DMSWARM`
  to be `npoints[0]` x `npoints[1]` (2D) or `npoints[0]` x `npoints[1]` x `npoints[2]` (3D). When using mode = `ADD_VALUES`,
  new points will be appended to any already existing in the `DMSWARM`

@*/
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)
{
  DMSwarmCellDM     celldm;
  DM                da = NULL;
  Vec               pos = NULL;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  PetscInt          d,i,j,k,p,Nfc,dim,n_estimate,n_found,n_new_est,n_curr;
  PetscInt          _npoints[] = {0, 0, 1};
  PetscInt          *swarm_cellid;
  PetscReal         O[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         L[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscReal         dx[3];
  PetscReal         *_pos,*swarm_coor;
  const PetscReal   *_coor;
  PetscErrorCode    (*coordFn)(PetscReal*, PetscReal*) = NULL;
  const char        **coordFields,*cellid;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  DMSWARMPICVALID(sw);
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);

  PetscCall(DMSwarmCellDMGetDM(celldm,&da));
  PetscCall(DMGetLocalBoundingBox(da,O,L));
  PetscCall(DMGetCoordinateDim(da, &dim));

  switch (dim) {
    case 2:
      coordFn = PointSetPolarToCartesianCoordinate;  
      break;
    
    case 3:
      coordFn = PointSetSphericalToCartesianCoordinate;
      break;
    
    default:
      SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_OUTOFRANGE, "Dimension can only be 2 or 3. Found: %" PetscInt_FMT, dim);
      break;
  }

  for (d=0; d<dim; d++) {
    if (npoints[d] > 1) {
      dx[d] = (max[d] - min[d]) / ((PetscReal)(npoints[d] - 1));
    } else {
      dx[d] = 0.0;
    }
    _npoints[d] = npoints[d];
  }

  /* determine number of points living in the bounding box */
  n_estimate = 0;
  for (k=0; k<_npoints[2]; k++) {
    for (j=0; j<_npoints[1]; j++) {
      for (i=0; i<_npoints[0]; i++) {
        PetscReal params[] = {0.0, 0.0, 0.0};
        PetscReal xp[] = {0.0, 0.0, 0.0};
        PetscInt  ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        /* Get the parameters r, theta (and phi if 3D) */
        for (d=0; d<dim; d++) { params[d] = min[d] + ijk[d] * dx[d]; }
        /* Compute the cartesian coordinates */
        PetscCall(coordFn(params, xp)); 
        /* Check if the points is in the DMDA */
        for (d=0; d<dim; d++) {
          if (xp[d] < O[d]) point_inside = PETSC_FALSE;
          if (xp[d] > L[d]) point_inside = PETSC_FALSE;
        }
        if (point_inside) n_estimate++;
      }
    }
  }

  /* create candidate list */
  PetscCall(VecCreate(PETSC_COMM_SELF, &pos));
  PetscCall(VecSetSizes(pos, dim * n_estimate, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(pos, dim));
  PetscCall(VecSetFromOptions(pos));
  PetscCall(VecGetArray(pos, &_pos));

  n_estimate = 0;
  for (k=0; k<_npoints[2]; k++) {
    for (j=0; j<_npoints[1]; j++) {
      for (i=0; i<_npoints[0]; i++) {
        PetscReal params[] = {0.0, 0.0, 0.0};
        PetscReal xp[] = {0.0, 0.0, 0.0};
        PetscInt  ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        /* Get the parameters r, theta (and phi if 3D) */
        for (d=0; d<dim; d++) { params[d] = min[d] + ijk[d] * dx[d]; }
        /* Compute the cartesian coordinates */
        PetscCall(coordFn(params, xp));
        /* Check if the points is in the DMDA */
        for (d=0; d<dim; d++) {
          if (xp[d] < O[d]) point_inside = PETSC_FALSE;
          if (xp[d] > L[d]) point_inside = PETSC_FALSE;
        }
        if (point_inside) {
          for (d=0; d<dim; d++) _pos[dim * n_estimate + d] = xp[d];
          n_estimate++;
        }
      }
    }
  }
  PetscCall(VecRestoreArray(pos,&_pos));

  /* locate points */
  PetscCall(DMLocatePoints(da,pos,DM_POINTLOCATION_NONE,&sfcell));
  PetscCall(PetscSFGetGraph(sfcell,NULL,NULL,NULL,&LA_sfcell));
  n_found = 0;
  for (p=0; p<n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) n_found++;
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    PetscCall(DMSwarmGetLocalSize(sw,&n_curr));
    n_new_est = n_curr + n_found;
    PetscCall(DMSwarmSetLocalSizes(sw,n_new_est,-1));
  }
  if (mode == INSERT_VALUES) {
    n_curr    = 0;
    n_new_est = n_found;
    PetscCall(DMSwarmSetLocalSizes(sw,n_new_est,-1));
  }

  /* initialize new coords, cell owners, pid */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellid));
  PetscCall(VecGetArrayRead(pos,&_coor));
  PetscCall(DMSwarmGetField(sw,coordFields[0],NULL,NULL,(void **)&swarm_coor));
  PetscCall(DMSwarmGetField(sw,cellid,NULL,NULL,(void **)&swarm_cellid));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (d=0; d<dim; d++) swarm_coor[dim * (n_curr + n_found) + d] = PetscRealPart(_coor[dim * p + d]);
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(VecRestoreArrayRead(pos, &_coor));

  PetscCall(PetscSFDestroy(&sfcell));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(PETSC_SUCCESS);
}