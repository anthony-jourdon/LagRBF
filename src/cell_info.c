#include <petsc.h>
#include "cell_info.h"

PetscLogEvent LAGRBF_CellInfoSetUp;

PetscErrorCode CellInfoCreate_SingleCell(CellInfo *ci)
{
  CellInfo c;
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&c));
  c->idx          = -1; // Cell index
  c->nneigh       = 0;  // Number of neighbours
  c->neighbours   = NULL; // Neighbour cells indices
  c->ndata        = 0; // Number of data points connected to the cell (include neighbours)
  c->data_idx     = NULL; // Data points indices connected to the cell (include neighbours)
  c->node_stencil = NULL; // Node stencil indices for the cell
  *ci = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoDestroy_SingleCell(CellInfo *ci)
{
  CellInfo c = *ci;
  PetscFunctionBeginUser;
  if (!c) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (c->neighbours)   { PetscCall(PetscFree(c->neighbours)); }
  if (c->data_idx)     { PetscCall(PetscFree(c->data_idx)); }
  if (c->node_stencil) { PetscCall(PetscFree(c->node_stencil)); }
  PetscCall(PetscFree(c));
  *ci = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoCreate(PartitionOfUnity pu, CellInfo **cellinfo)
{
  DM             da = NULL;
  CellInfo       *ci = NULL;
  PetscInt       iel,nel,nen;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Get the local sequential DMDA */
  PetscCall(PUGetDMDALocal(pu,&da));
  /* Get the elements */
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  PetscCall(PetscCalloc1(nel,&ci));
  for (iel=0; iel<nel; iel++) { PetscCall(CellInfoCreate_SingleCell(&ci[iel])); }
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscCall(CellInfoCreateNeighbourElements(pu,ci));
  //PetscCall(CellInfoCreateNodeStencils(pu,ci));
  *cellinfo = ci;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoDestroy(PartitionOfUnity pu, CellInfo **cellinfo)
{
  DM             da = NULL;
  CellInfo       *ci = *cellinfo;
  PetscInt       iel,nel,nen;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  if (!ci) { PetscFunctionReturn(PETSC_SUCCESS); }
  /* Get the local sequential DMDA */
  PetscCall(PUGetDMDALocal(pu,&da));
  /* Get the elements */
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  for (iel=0; iel<nel; iel++) { PetscCall(CellInfoDestroy_SingleCell(&ci[iel])); }
  PetscCall(PetscFree(ci));
  *cellinfo = NULL;
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoSetUp(PartitionOfUnity pu, CellInfo *ci, DM swarm)
{
  DM             da = NULL;
  PetscInt       iel,nel,nen;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscLogEventBegin(LAGRBF_CellInfoSetUp,0,0,0,0);
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  PetscCall(DMSwarmSortGetAccess(swarm));
  for (iel=0; iel<nel; iel++) {
    PetscCall(CellInfoFillDataIdx(ci[iel],swarm));
  }
  PetscCall(DMSwarmSortRestoreAccess(swarm));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscLogEventEnd(LAGRBF_CellInfoSetUp,0,0,0,0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CellInfoCreateNeighbourElements_2D(CellInfo *ci, PetscInt start[], PetscInt size[])
{
  PetscInt i,j;
  PetscFunctionBeginUser;
  for (j=0; j<size[1]; j++) {
    for (i=0; i<size[0]; i++) {
      CellInfo cell_info;
      PetscInt eidx;
      PetscInt *neighbours;
      /* Cell index */
      eidx = i + j*size[0];
      cell_info = ci[eidx];
      cell_info->idx    = eidx;
      cell_info->nneigh = 9;
      PetscCall(PetscMalloc1(9,&neighbours));

      neighbours[0] = eidx - size[0] - 1; // Bottom-left
      neighbours[1] = eidx - size[0];     // Bottom
      neighbours[2] = eidx - size[0] + 1; // Bottom-right
      neighbours[3] = eidx - 1;           // Left
      neighbours[4] = eidx;               // Center
      neighbours[5] = eidx + 1;           // Right
      neighbours[6] = eidx + size[0] - 1; // Top-left
      neighbours[7] = eidx + size[0];     // Top
      neighbours[8] = eidx + size[0] + 1; // Top-right
      /* Mark invalid neighbours */
      if (i == 0) {
        neighbours[0] = -1; // Bottom-left
        neighbours[3] = -1; // Left
        neighbours[6] = -1; // Top-left
      }
      if (i == size[0]-1) {
        neighbours[2] = -1; // Bottom-right
        neighbours[5] = -1; // Right
        neighbours[8] = -1; // Top-right
      }
      if (j == 0) {
        neighbours[0] = -1; // Bottom-left
        neighbours[1] = -1; // Bottom
        neighbours[2] = -1; // Bottom-right
      }
      if (j == size[1]-1) {
        neighbours[6] = -1; // Top-left
        neighbours[7] = -1; // Top
        neighbours[8] = -1; // Top-right
      }
      cell_info->neighbours = neighbours;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CellInfoCreateNeighbourElements_3D(CellInfo *ci, PetscInt start[], PetscInt size[])
{
  PetscInt i,j,k;
  PetscFunctionBeginUser;
  for (k=0; k<size[2]; k++) {
    for (j=0; j<size[1]; j++) {
      for (i=0; i<size[0]; i++) {
        PetscInt eidx;
        PetscInt *neighbours;
        CellInfo cell_info;
        /* Cell index */
        eidx = i + j*size[0] + k*size[0]*size[1];
        cell_info = ci[eidx];
        cell_info->idx    = eidx;
        cell_info->nneigh = 27;
        PetscCall(PetscMalloc1(27,&neighbours));
        
        // Bottom front
        neighbours[0] = eidx - size[0]*size[1] - size[0] - 1; // Bottom-left-front
        neighbours[1] = eidx - size[0]*size[1] - size[0];     // Bottom-front
        neighbours[2] = eidx - size[0]*size[1] - size[0] + 1; // Bottom-right-front
        // Middle front
        neighbours[3] = eidx - size[0]*size[1] - 1;           // Left-front
        neighbours[4] = eidx - size[0]*size[1];               // Center-front
        neighbours[5] = eidx - size[0]*size[1] + 1;           // Right-front
        // Top front
        neighbours[6] = eidx - size[0]*size[1] + size[0] - 1; // Top-left-front
        neighbours[7] = eidx - size[0]*size[1] + size[0];     // Top-front
        neighbours[8] = eidx - size[0]*size[1] + size[0] + 1; // Top-right-front
        // Bottom centre
        neighbours[9 ] = eidx - size[0] - 1;                 // Bottom-left
        neighbours[10] = eidx - size[0];                     // Bottom
        neighbours[11] = eidx - size[0] + 1;                 // Bottom-right
        // Middle centre
        neighbours[12] = eidx - 1;                           // Left
        neighbours[13] = eidx;                               // Center
        neighbours[14] = eidx + 1;                           // Right
        // Top centre
        neighbours[15] = eidx + size[0] - 1;                 // Top-left
        neighbours[16] = eidx + size[0];                     // Top
        neighbours[17] = eidx + size[0] + 1;                 // Top-right
        // Bottom back
        neighbours[18] = eidx + size[0]*size[1] - size[0] - 1; // Bottom-left-back
        neighbours[19] = eidx + size[0]*size[1] - size[0];     // Bottom-back
        neighbours[20] = eidx + size[0]*size[1] - size[0] + 1; // Bottom-right-back
        // Middle back
        neighbours[21] = eidx + size[0]*size[1] - 1;           // Left-back
        neighbours[22] = eidx + size[0]*size[1];               // Center-back
        neighbours[23] = eidx + size[0]*size[1] + 1;           // Right-back
        // Top back
        neighbours[24] = eidx + size[0]*size[1] + size[0] - 1; // Top-left-back
        neighbours[25] = eidx + size[0]*size[1] + size[0];     // Top-back
        neighbours[26] = eidx + size[0]*size[1] + size[0] + 1; // Top-right-back
        /* Mark invalid neighbours */
        if (i == 0) {
          neighbours[ 0] = -1; // Bottom-left-front
          neighbours[ 3] = -1; // Left-front
          neighbours[ 6] = -1; // Top-left-front
          neighbours[ 9] = -1; // Bottom-left
          neighbours[12] = -1; // Left
          neighbours[15] = -1; // Top-left
          neighbours[18] = -1; // Bottom-left-back
          neighbours[21] = -1; // Left-back
          neighbours[24] = -1; // Top-left-back
        }
        if (i == size[0]-1) {
          neighbours[ 2] = -1; // Bottom-right-front
          neighbours[ 5] = -1; // Right-front
          neighbours[ 8] = -1; // Top-right-front 
          neighbours[11] = -1; // Bottom-right
          neighbours[14] = -1; // Right
          neighbours[17] = -1; // Top-right
          neighbours[20] = -1; // Bottom-right-back
          neighbours[23] = -1; // Right-back
          neighbours[26] = -1; // Top-right-back
        }
        if (j == 0) {
          neighbours[ 0] = -1; // Bottom-left-front
          neighbours[ 1] = -1; // Bottom-front
          neighbours[ 2] = -1; // Bottom-right-front
          neighbours[ 9] = -1; // Bottom-left
          neighbours[10] = -1; // Bottom  
          neighbours[11] = -1; // Bottom-right
          neighbours[18] = -1; // Bottom-left-back
          neighbours[19] = -1; // Bottom-back
          neighbours[20] = -1; // Bottom-right-back
        }
        if (j == size[1]-1) {
          neighbours[ 6] = -1; // Top-left-front
          neighbours[ 7] = -1; // Top-front
          neighbours[ 8] = -1; // Top-right-front
          neighbours[15] = -1; // Top-left
          neighbours[16] = -1; // Top
          neighbours[17] = -1; // Top-right
          neighbours[24] = -1; // Top-left-back
          neighbours[25] = -1; // Top-back
          neighbours[26] = -1; // Top-right-back
        }
        if (k == 0) {
          neighbours[0] = -1; // Bottom-left-front
          neighbours[1] = -1; // Bottom-front
          neighbours[2] = -1; // Bottom-right-front
          neighbours[3] = -1; // Left-front
          neighbours[4] = -1; // Center-front
          neighbours[5] = -1; // Right-front
          neighbours[6] = -1; // Top-left-front
          neighbours[7] = -1; // Top-front
          neighbours[8] = -1; // Top-right-front
        }
        if (k == size[2]-1) {
          neighbours[18] = -1; // Bottom-left-back
          neighbours[19] = -1; // Bottom-back
          neighbours[20] = -1; // Bottom-right-back
          neighbours[21] = -1; // Left-back
          neighbours[22] = -1; // Center-back
          neighbours[23] = -1; // Right-back
          neighbours[24] = -1; // Top-left-back
          neighbours[25] = -1; // Top-back
          neighbours[26] = -1; // Top-right-back
        }
        cell_info->neighbours = neighbours;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoCreateNeighbourElements(PartitionOfUnity pu, CellInfo *ci)
{
  DM             da = NULL;
  PetscInt       dim;
  PetscInt       start[3],size[3];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMGetDimension(da,&dim));
  /* Get elements corners */
  PetscCall(DMDAGetElementsCorners(da, &start[0], &start[1], &start[2]));
  PetscCall(DMDAGetElementsSizes(da, &size[0], &size[1], &size[2]));

  switch (dim) {
  case 2:
    PetscCall(CellInfoCreateNeighbourElements_2D(ci,start,size));
    break;

  case 3:
    PetscCall(CellInfoCreateNeighbourElements_3D(ci,start,size));
    break;
  
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 2 or 3. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CellInfoNodeStencil_2D(CellInfo *ci, DM da)
{
  PetscInt i,j;
  PetscInt size[2];
  PetscFunctionBeginUser;
  PetscCall(DMDAGetElementsSizes(da,&size[0],&size[1],NULL));
  for (j=0; j<size[1]; j++) {
    for (i=0; i<size[0]; i++) {
      CellInfo cell_info;
      PetscInt eidx,start_eidx,start_nidx,ii,jj,sw;
      PetscInt leidx[2],lseidx[2],stencil_width[] = {4,4};
      PetscInt *stencil;
      /* Cell index */
      eidx      = i + j*size[0];
      cell_info = ci[eidx];
      /* Allocate the array to hold the stencil */
      sw = stencil_width[0]*stencil_width[1];
      PetscCall(PetscMalloc1(sw,&stencil));
      for (ii=0; ii<sw; ii++) { stencil[ii] = -1; } // Initialize to invalid
      /* Get the logical indices (i,j,k) of the element */
      PetscCall(DMDAGetElementLogicalIndex(da,eidx,&leidx[0],&leidx[1],NULL));
      /* Ensure that we do not count nodes and elements outside the domain */
      if (leidx[0] == 0) {
        leidx[0] = 1;
        stencil_width[0] -= 1;
      }
      if (leidx[1] == 0) {
        leidx[1] = 1;
        stencil_width[1] -= 1;
      }
      if (leidx[0] == size[0]-1) {
        stencil_width[0] -= 1;
      }
      if (leidx[1] == size[1]-1) {
        stencil_width[1] -= 1;
      }
      start_eidx = (leidx[0]-1) + (leidx[1]-1)*size[0];
      PetscCall(DMDAGetElementLogicalIndex(da,start_eidx,&lseidx[0],&lseidx[1],NULL));
      start_nidx = start_eidx + lseidx[1];
      /* Fill the stencil */
      for (jj=0; jj<stencil_width[1]; jj++) {
        for (ii=0; ii<stencil_width[0]; ii++) {
          PetscInt nidx = start_nidx + ii + jj*(size[0]+1);
          stencil[ ii + jj*stencil_width[0] ] = nidx;
        }
      }
      cell_info->node_stencil = stencil;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CellInfoNodeStencil_3D(CellInfo *ci, DM da)
{
  PetscInt i,j,k,sw;
  PetscInt size[3];
  PetscFunctionBeginUser;
  PetscCall(DMDAGetElementsSizes(da,&size[0],&size[1],&size[2]));
  for (k=0; k<size[2]; k++) {
    for (j=0; j<size[1]; j++) {
      for (i=0; i<size[0]; i++) {
        CellInfo cell_info;
        PetscInt eidx,start_eidx,start_nidx,ii,jj,kk;
        PetscInt leidx[3],lseidx[3],stencil_width[] = {4,4,4};
        PetscInt *stencil;
        /* Cell index */
        eidx      = i + j*size[0] + k*size[0]*size[1];
        cell_info = ci[eidx];
        /* Allocate the array to hold the stencil */
        sw = stencil_width[0]*stencil_width[1]*stencil_width[2];
        PetscCall(PetscMalloc1(sw,&stencil));
        for (ii=0; ii<sw; ii++) { stencil[ii] = -1; } // Initialize to invalid
        /* Get the logical indices (i,j,k) of the element */
        PetscCall(DMDAGetElementLogicalIndex(da,eidx,&leidx[0],&leidx[1],&leidx[2]));
        /* Ensure that we do not count nodes and elements outside the domain */
        if (leidx[0] == 0) {
          leidx[0] = 1;
          stencil_width[0] -= 1;
        }
        if (leidx[1] == 0) {
          leidx[1] = 1;
          stencil_width[1] -= 1;
        }
        if (leidx[2] == 0) {
          leidx[2] = 1;
          stencil_width[2] -= 1;
        }
        if (leidx[0] == size[0]-1) {
          stencil_width[0] -= 1;
        }
        if (leidx[1] == size[1]-1) {
          stencil_width[1] -= 1;
        }
        if (leidx[2] == size[2]-1) {
          stencil_width[2] -= 1;
        }
        start_eidx = (leidx[0]-1) + (leidx[1]-1)*size[0] + (leidx[2]-1)*size[0]*size[1];
        PetscCall(DMDAGetElementLogicalIndex(da,start_eidx,&lseidx[0],&lseidx[1],&lseidx[2]));
        start_nidx = start_eidx + lseidx[1] + lseidx[2]*(size[0]+1);
        /* Fill the stencil */
        for (kk=0; kk<stencil_width[2]; kk++) {
          for (jj=0; jj<stencil_width[1]; jj++) {
            for (ii=0; ii<stencil_width[0]; ii++) {
              PetscInt nidx = start_nidx + ii + jj*(size[0]+1) + kk*(size[0]+1)*(size[1]+1);
              stencil[ ii + jj*stencil_width[0] + kk*stencil_width[0]*stencil_width[1] ] = nidx;
            }
          }
        }
        cell_info->node_stencil = stencil;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoCreateNodeStencils(PartitionOfUnity pu, CellInfo *ci)
{
  DM             da = NULL;
  PetscInt       dim;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMGetDimension(da,&dim));

  switch (dim) {
  case 2:
    PetscCall(CellInfoNodeStencil_2D(ci,da));
    break;

  case 3:
    PetscCall(CellInfoNodeStencil_3D(ci,da));
    break;
  
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 2 or 3. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoFillDataIdx(CellInfo ci, DM swarm)
{
  PetscInt iel,eidx,nneigh,offset;
  PetscInt *neighbours = NULL;
  PetscInt *point_count;
  PetscInt **point_idx;
  PetscFunctionBeginUser;
  PetscCall(CellInfoGetIndex(ci,&eidx));
  nneigh = -1; // Initial invalid value
  PetscCall(CellInfoGetNeighbours(ci,&nneigh,&neighbours));
  /* Create 2 arrays to store points and indices per element */
  PetscCall(PetscCalloc2(nneigh,&point_count,nneigh,&point_idx));
  ci->ndata = 0;
  for (iel=0; iel<nneigh; iel++) {
    PetscInt neidx = neighbours[iel];
    if (neidx < 0) { continue; } // Skip invalid neighbours
    PetscCall(DMSwarmSortGetPointsPerCell(swarm,neidx,&point_count[iel],&point_idx[iel]));
    ci->ndata += point_count[iel];
  }
  /* Allocate the data indices array */
  PetscCall(PetscMalloc1(ci->ndata,&ci->data_idx));
  /* Populate */
  offset = 0;
  for (iel=0; iel<nneigh; iel++) {
    PetscInt neidx = neighbours[iel];
    if (neidx < 0) { continue; } // Skip invalid neighbours
    /* Copy the points indices to the data indices array */
    memcpy(&ci->data_idx[offset],point_idx[iel],point_count[iel] * sizeof(PetscInt)); 
    offset += point_count[iel]; // Update the offset for the next cell
    PetscCall(DMSwarmSortRestorePointsPerCell(swarm,neidx,&point_count[iel],&point_idx[iel]));
  }
  PetscCall(PetscFree2(point_count,point_idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoReset(PartitionOfUnity pu, CellInfo *ci)
{
  DM             da;
  PetscInt       iel,nel,nen;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscCall(PUGetDMDALocal(pu,&da));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  for (iel=0; iel<nel; iel++) {
    CellInfo cell_info = ci[iel];
    /* Reset data indices */
    if (cell_info->data_idx) {
      PetscCall(PetscFree(cell_info->data_idx));
      cell_info->data_idx = NULL;
      cell_info->ndata    = 0;
    }
  }
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoGetIndex(CellInfo ci, PetscInt *idx)
{
  PetscFunctionBeginUser;
  PetscCheck(ci,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"CellInfo structure is NULL. Call CellInfoCreate() first.");
  if (idx) { *idx = ci->idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoGetNumberOfNeighbours(CellInfo ci, PetscInt *nneigh)
{
  PetscFunctionBeginUser;
  PetscCheck(ci,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"CellInfo structure is NULL. Call CellInfoCreate() first.");
  if (nneigh) { *nneigh = ci->nneigh; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoGetNeighbours(CellInfo ci, PetscInt *nneigh, PetscInt **neighb)
{
  PetscFunctionBeginUser;
  PetscCheck(ci,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"CellInfo structure is NULL. Call CellInfoCreate() first.");
  PetscCheck(ci->neighbours,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CellInfo neighbours array not created yet. Call CellInfoCreateNeighbourElements() first.");
  if (nneigh) { *nneigh = ci->nneigh; }
  if (neighb) { *neighb = ci->neighbours; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoGetDataPointsIdx(CellInfo ci, PetscInt *ndata, PetscInt **data_idx)
{
  PetscFunctionBeginUser;
  PetscCheck(ci,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"CellInfo structure is NULL. Call CellInfoCreate() first.");
  if (ndata)                      { *ndata = ci->ndata; }
  if (ci->ndata == 0 && data_idx) { *data_idx = NULL;}
  else if (data_idx)              { *data_idx = ci->data_idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDAGetElementLogicalIndex(DM da, PetscInt eidx, PetscInt *ix, PetscInt *iy, PetscInt *iz)
{
  PetscInt dim,i,j,k;
  PetscInt size[3];
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(da,&dim));
  PetscCall(DMDAGetElementsSizes(da,&size[0],&size[1],&size[2]));
  i = j = k = -1;
  switch (dim) {
    case 2:
      j = (PetscInt)(eidx / size[0]); // j
      i = (PetscInt)(eidx - j*size[0]); // i
      break;
    
    case 3:
      k = (PetscInt)(eidx / (size[0]*size[1])); // k
      j = (PetscInt)((eidx - k*size[0]*size[1]) / size[0]); // j
      i = (PetscInt)(eidx - k*size[0]*size[1] - j*size[0]); // i
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 2 or 3. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
      break;
  }
  if (ix) { *ix = i; }
  if (iy) { *iy = j; }
  if (iz) { *iz = k; }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellInfoGetStencilNodes(CellInfo ci, PetscInt **stencil)
{
  PetscFunctionBeginUser;
  PetscCheck(ci,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"CellInfo structure is NULL. Call CellInfoCreate() first.");
  PetscCheck(ci->node_stencil,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CellInfo node stencil array not created yet. Call CellInfoCreateNodeStencils() first.");
  if (stencil) { *stencil = ci->node_stencil; }
  PetscFunctionReturn(PETSC_SUCCESS);
}