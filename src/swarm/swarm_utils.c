#include <petsc.h>
#include <petsc/private/dmswarmimpl.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "swarm/swarm_utils.h"
#include "io_utils.h"

PetscLogEvent LAGRBF_SphericalCoordinates_Halton;

PetscErrorCode CopyContiguousBlockData_Generic(
  void      *dest,        // destination array
  void      *source,      // source array
  PetscInt  npoints,     // number of points
  PetscInt  nblocks,     // number of blocks to copy
  PetscInt  bs,          // block size of the source array
  PetscInt  start_block, // starting block index in the source array
  size_t    typesize     // size of the data type (e.g., sizeof(PetscReal) or sizeof(PetscInt))
)
{
  PetscInt np;
  unsigned char *d = (unsigned char *)dest;
  unsigned char *s = (unsigned char *)source;
  PetscFunctionBeginUser;
  PetscCheck(nblocks > 0 && nblocks <= bs && start_block >= 0 && start_block + nblocks <= bs,
           PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
           "Invalid parameters: nblocks=%"PetscInt_FMT", bs=%"PetscInt_FMT", start_block=%"PetscInt_FMT"", nblocks, bs, start_block);
  for (np=0; np<npoints; np++) {
    memcpy(&d[nblocks*np*typesize], &s[(bs*np + start_block)*typesize], nblocks*typesize);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopyContiguousBlockData_Real(
  PetscReal *dest,        // destination array
  PetscReal *source,      // source array
  PetscInt  npoints,     // number of points
  PetscInt  nblocks,     // number of blocks to copy
  PetscInt  bs,          // block size of the source array
  PetscInt  start_block  // starting block index in the source array
)
{
  PetscInt np;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  for (np=0; np<npoints; np++) {
    PetscArraycpy(&dest[nblocks*np], &source[(bs*np + start_block)], nblocks);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopyArbitraryBlockData_Real(
  PetscReal dest[],          // destination array
  PetscReal source[],        // source array
  PetscInt  npoints,         // number of points
  PetscInt  nblocks,         // number of blocks to copy
  PetscInt  bs,              // block size of the source array
  PetscInt  blocks_to_copy[] // array of block indices to copy from the source array
)
{
  PetscInt np,b;
  PetscFunctionBeginUser;
  for (np=0; np<npoints; np++) {
    for (b=0; b<nblocks; b++) {
      PetscInt block_id = blocks_to_copy[b];
      /* copy point */
      dest[nblocks*np + b] = source[bs*np + block_id];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

PetscReal halton_sequence(PetscInt index, PetscInt base)
{
  PetscReal f = 1.0, r = 0.0;
  PetscFunctionBeginUser;
  while (index > 0) {
    f = f / (PetscReal)base;
    r = r + f * (PetscReal)(index % base);
    index = index / base;
  }
  PetscFunctionReturn(r);
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
  DMSwarmSetPointsSphericalCoordinates_Halton - Set point coordinates in a `DMSWARM` on a regular (ijk) spherical grid

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
static PetscErrorCode DMSwarmSetPointsSphericalCoordinates_Uniform(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)
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

static PetscErrorCode PolarParameters_Halton(
  PetscInt pidx,
  PetscReal min[], 
  PetscReal max[], 
  PetscReal dx[],
  PetscReal params[]
)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = halton_sequence(pidx+1,2); // base 2 can be changed
  params[0] = PetscSqrtReal( ( max[0]*max[0] - min[0]*min[0] )*u + min[0]*min[0] );
  params[1] = min[1] + pidx * dx[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SphericalParameters_Halton(
  PetscInt pidx,
  PetscReal min[], 
  PetscReal max[], 
  PetscReal dx[],
  PetscReal params[]
)
{
  PetscReal u,v;
  PetscFunctionBeginUser;
  u = halton_sequence(pidx+1,2);
  v = halton_sequence(pidx+1,3);
  params[0] = PetscSqrtReal( ( max[0]*max[0] - min[0]*min[0] )*u + min[0]*min[0] ); // R
  params[1] = PetscAcosReal(1.0 - 2.0*v); // theta
  params[2] = min[2] + pidx * dx[2];  // phi
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetPointsSphericalCoordinates_Halton - Set point coordinates in a `DMSWARM` on a spherical grid using Halton sequence

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
static PetscErrorCode DMSwarmSetPointsSphericalCoordinates_Halton(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)
{
  DMSwarmCellDM     celldm;
  DM                da = NULL;
  Vec               pos = NULL;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  PetscInt          i,j,k,p,n_estimate,d,Nfc,dim,n_found,n_new_est,n_curr;
  PetscInt          _npoints[] = {0, 0, 1};
  PetscInt          *swarm_cellid;
  PetscReal         dx[3];
  PetscReal         O[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         L[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscReal         *_pos,*swarm_coor;
  const PetscReal   *_coor;
  const char        **coordFields,*cellid;
  PetscErrorCode    (*coordFn)(PetscReal*, PetscReal*) = NULL;
  PetscErrorCode    (*paramsFn)(PetscInt,PetscReal*,PetscReal*,PetscReal*,PetscReal*) = NULL;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PetscLogEventBegin(LAGRBF_SphericalCoordinates_Halton,0,0,0,0));
  DMSWARMPICVALID(sw);
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);

  PetscCall(DMSwarmCellDMGetDM(celldm,&da));
  PetscCall(DMGetLocalBoundingBox(da,O,L));
  PetscCall(DMGetCoordinateDim(da, &dim));

  /* Set coordinate and parameter functions based on dimension */
  switch (dim) {
    case 2:
      coordFn  = PointSetPolarToCartesianCoordinate;
      paramsFn = PolarParameters_Halton;
      break;
    
    case 3:
      coordFn  = PointSetSphericalToCartesianCoordinate;
      paramsFn = SphericalParameters_Halton;
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
        PetscReal xp[]     = {0.0, 0.0, 0.0};
        PetscInt  pidx;
        PetscBool point_inside = PETSC_TRUE;

        pidx   = i + _npoints[0]*j + _npoints[0]*_npoints[1]*k;
        /* Get the parameters r, theta (and phi if 3D) */
        PetscCall(paramsFn(pidx, min, max, dx, params));
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
        PetscInt  pidx;
        PetscBool point_inside = PETSC_TRUE;

        pidx   = i + _npoints[0]*j + _npoints[0]*_npoints[1]*k;
        /* Get the parameters r, theta (and phi if 3D) */
        PetscCall(paramsFn(pidx, min, max, dx, params));
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
  PetscCall(PetscLogEventEnd(LAGRBF_SphericalCoordinates_Halton,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetPointsSphericalCoordinates - Set point coordinates in a `DMSWARM` on a spherical grid

  Collective

  Input Parameters:
+ sw      - the `DMSWARM`
. min     - minimum coordinate values in the r, theta, phi directions (array of length dim)
. max     - maximum coordinate values in the r, theta, phi directions (array of length dim)
. npoints - number of points in each spatial direction (array of length dim)
. mode    - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)
- halton  - if `PETSC_TRUE`, use Halton sequence to distribute points, otherwise use uniform distribution

  Notes:
  When using mode = `INSERT_VALUES`, this method will reset the number of particles in the `DMSWARM`
  to be `npoints[0]` x `npoints[1]` (2D) or `npoints[0]` x `npoints[1]` x `npoints[2]` (3D). When using mode = `ADD_VALUES`,
  new points will be appended to any already existing in the `DMSWARM`

@*/
PetscErrorCode DMSwarmSetPointsSphericalCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode, PetscBool halton)
{
  PetscFunctionBeginUser;
  if (halton) {
    PetscCall(DMSwarmSetPointsSphericalCoordinates_Halton(sw, min, max, npoints, mode));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMSwarmSetPointsSphericalCoordinates_Uniform(sw, min, max, npoints, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}