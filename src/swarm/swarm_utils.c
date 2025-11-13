#include <petsc.h>
#include <petsc/private/dmswarmimpl.h>
#include "lagrangian_rbf.h"
#include "partition_of_unity.h"
#include "swarm/swarm_utils.h"
#include "io_utils.h"

PetscLogEvent LAGRBF_SphericalCoordinates_Halton;

PetscErrorCode SwarmFieldAssignUniqueIndex(DM sw, const char *field_name, PetscInt start_idx, PetscInt end_idx)
{
  PetscInt      p,npoints,bs;
  PetscInt64    np_local,max_local,max,np_global;
  PetscInt64    *pidx;
  PetscMPIInt   rank;
  PetscDataType dtype;
  MPI_Comm      comm;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PetscObjectGetComm((PetscObject)sw,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetField(sw, field_name, &bs, &dtype, (void **)&pidx));
  PetscCheck(dtype == PETSC_INT64, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Field %s must be of type PETSC_INT64", field_name);
  PetscCheck(bs == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Field %s must be a scalar of blocksize 1, found: %" PetscInt_FMT, field_name, bs);
  /* find max pid presently in the system */
  max_local = 0;
  for (p=0; p<npoints; p++) {
    PetscInt64 mp_idx = pidx[p]; 
    if ( mp_idx > max_local ) { max_local = mp_idx; }
  }
  PetscCallMPI(MPI_Allreduce(&max_local,&max,1,MPIU_INT64,MPI_MAX,comm));
  /*PetscPrintf(comm,"[[ %s ]] : max_pid = %" PetscInt_FMT "\n",PETSC_FUNCTION_NAME,max);*/
  max += 1;

  /* give particles a unique identifier */
  np_local = (PetscInt64)(end_idx - start_idx);
  PetscCallMPI(MPI_Scan(&np_local,&np_global,1,MPIU_INT64,MPI_SUM,comm));
  //printf("rank %d : np_local = %" PetscInt_FMT ", np_global = %" PetscInt_FMT "\n",(int)rank,np_local,np_global);
  for (p=start_idx; p<end_idx; p++) {
    pidx[p] = max + (np_global-np_local) + (PetscInt64)(p-start_idx);
    //printf("assigning %" PetscInt_FMT " -> pid = %" PetscInt_FMT " \n", p, pidx[p] );
  }
  PetscCall(DMSwarmRestoreField(sw, field_name, &bs, &dtype, (void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

        pidx = i + _npoints[0]*j + _npoints[0]*_npoints[1]*k;
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
          memcpy(&_pos[dim * n_estimate], xp, dim * sizeof(PetscReal));
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
      memcpy(&swarm_coor[dim * (n_curr + n_found)], &_coor[dim * p], dim * sizeof(PetscReal));
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

/*
To remove the points at the specified indices from the swarm, 
We move all points that are not being removed to the front of the swarm,
and then we resize the swarm to exclude the removed points.
*/

PetscErrorCode DMSwarmRemoveNPointsFromIndex(DM swarm, PetscInt nindices, PetscInt *indices)
{
  PetscInt  i,np,npoints;
  PetscInt  next_rm,dst_idx;
  PetscBool sorted;
  PetscFunctionBeginUser;
  
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCheck(nindices <= npoints, PetscObjectComm((PetscObject)swarm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot remove %" PetscInt_FMT " points from %" PetscInt_FMT " total points", nindices, npoints);
  /* Check that no index in indices is out of range */
  for (i=0; i<nindices; i++) {
    PetscCheck(indices[i] >=0 && indices[i] < npoints, PetscObjectComm((PetscObject)swarm), PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is out of range [0, %" PetscInt_FMT ")", indices[i], npoints);
  }

  /* If indices are not sorted, we sort them */
  PetscCall(PetscSortedInt(nindices, indices, &sorted));
  if (!sorted) { PetscCall(PetscSortInt(nindices, indices)); }

  next_rm = 0;
  dst_idx = 0;
  for (np=0; np<npoints; np++) {
    if (next_rm < nindices && np == indices[next_rm]) {
      next_rm++;
    } else {
      if (dst_idx != np) {
        /* copy point np to dst_idx */
        PetscCall(DMSwarmCopyPoint(swarm, np, dst_idx));
      }
      dst_idx++;
    }
  }
  /* Now we remove the last nindices points from the swarm */
  PetscCall(DMSwarmSetLocalSizes(swarm, dst_idx, -1));

  PetscFunctionReturn(PETSC_SUCCESS);
}