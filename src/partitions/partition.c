#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "kernels/kernels_rbf.h"

/*
  Create the inverse map from partition to cells.
  The partition is a node of the DMDA, and the cells are the elements of the DMDA.
  The map is an array of indices, where each index corresponds to a cell that is connected to the partition.
  The number of cells connected to each partition is stored in ncells.
  A map array is allocated for each partition and filled with the cell indices.
  Memory is freed in PartitionDestroy_SinglePartition() for a given partition or in PartitionDestroy() for all partitions.
*/
PetscErrorCode PartitionCreate_PartitionToCellInverseMapping(PartitionOfUnity pu, Partition *partition)
{
  DM             das;
  PetscInt       iel,k,nel,nen,nidx;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));

  /* Count the number of cells connected to each partition */
  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      nidx = elidx[nen*iel + k]; // Partition index from the cell index
      partition[nidx]->ncells++; // count the number of elements that contain the vertex
    }
  }
  for (k=0; k<pu->npart; k++) {
    /* Allocate the inverse map */
    PetscCall(PetscMalloc1(partition[k]->ncells,&partition[k]->map));
    /* Reset the counter */
    partition[k]->ncells = 0;
  }
  /* Fill the inverse map */
  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      nidx = elidx[nen*iel + k]; // partition index from the cell index
      partition[nidx]->map[ partition[nidx]->ncells ] = iel; // store the element index
      partition[nidx]->ncells++; // increment the counter
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionsSetIndex_1D(DM da, Partition *partitions)
{
  PetscInt i,dim,si,nx;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(da,&dim));
  PetscCheck(dim == 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 1. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
  PetscCall(DMDAGetCorners(da,&si,NULL,NULL,&nx,NULL,NULL));
  for (i=0; i<nx; i++) { partitions[i]->idx = i; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionsSetIndex_2D(DM da, Partition *partitions)
{
  PetscInt i,j,cnt,dim,si,sj,nx,ny;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(da,&dim));
  PetscCheck(dim == 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 2. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  cnt = 0;
  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      PetscInt puidx = i + j*nx;
      partitions[cnt]->idx = puidx; // Set the index of the partition
      cnt++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionsSetIndex_3D(DM da, Partition *partitions)
{
  PetscInt i,j,k,cnt,dim,si,sj,sk,nx,ny,nz;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(da,&dim));
  PetscCheck(dim == 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"[[ %s ]]: Spatial dimension must be 3. Found: %" PetscInt_FMT,PETSC_FUNCTION_NAME,dim);
  PetscCall(DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz));
  cnt = 0;
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        PetscInt puidx = i + j*nx + k*nx*ny;
        partitions[cnt]->idx = puidx; // Set the index of the partition
        cnt++;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionsSetIndex(PartitionOfUnity pu, Partition *partitions)
{
  DM da;
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME));
  PetscCall(PUGetDMDALocal(pu,&da));
  switch (pu->dim) {
    case 1:
      PetscCall(PartitionsSetIndex_1D(da, partitions));
      break;
    case 2:
      PetscCall(PartitionsSetIndex_2D(da, partitions));
      break;
    case 3:
      PetscCall(PartitionsSetIndex_3D(da, partitions));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Dimension can only be 1, 2 or 3. Found: %" PetscInt_FMT,pu->dim);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionCreate_SinglePartition(Partition *partition)
{
  Partition p;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&p));
  p->X        = NULL; // Matrix holding coefficients for the partition interpolation
  p->A        = NULL; // RBF matrix for the partition
  p->ncells   = 0; // Initialize number of cells connected to the partition
  p->map      = NULL; // Initialize the map
  p->ndata    = 0; // Initialize number of data points in the partition
  p->data_idx = NULL; // Initialize the data points indices
  p->idx      = -1; // Index of the partition
  p->setup    = PETSC_FALSE;
  p->solved   = PETSC_FALSE;
  p->data     = NULL;
  
  *partition = p;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionDestroy_SinglePartition(Partition *partition, PetscErrorCode (*destroy)(Partition))
{
  Partition p = *partition;
  PetscFunctionBeginUser;

  if (!p)          { PetscFunctionReturn(PETSC_SUCCESS); }
  if (p->map)      { PetscCall(PetscFree(p->map));       }
  if (p->data_idx) { PetscCall(PetscFree(p->data_idx));  }
  if (p->X)        { PetscCall(MatDestroy(&p->X));       }
  if (p->A)        { PetscCall(MatDestroy(&p->A));       }
  if (destroy)     { PetscCall(destroy(p));           }
  PetscCall(PetscFree(p));
  *partition = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionRBFMatClear(Partition p)
{
  PetscFunctionBeginUser;
  if (p->A) { PetscCall(MatDestroy(&p->A)); }
  p->A = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition)
{
  Partition *p;
  PetscInt  n,npart;
  //PetscReal epsilon;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscCall(PetscCalloc1(npart,&p));
  for (n=0; n<npart; n++) { PetscCall(PartitionCreate_SinglePartition(&p[n])); }
  /* Create the inverse map connecting partition to cells */
  PetscCall(PartitionCreate_PartitionToCellInverseMapping(pu,p));
  /* Set the index of each partition */
  PetscCall(PartitionsSetIndex(pu,p));
  /* Set the epsilon value for each partition */
  //PetscCall(PUGetEpsilon(pu,&epsilon));
  //for (n=0; n<npart; n++) { PetscCall(PartitionSetEpsilon(p[n],epsilon)); }

  *partition = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition)
{
  PetscInt n,npart;
  Partition *p = *partition;
  PetscFunctionBeginUser;

  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  for (n=0; n<npart; n++) { PetscCall(PartitionDestroy_SinglePartition(&p[n], pu->pdestroy)); }
  PetscCall(PetscFree(p));
  *partition = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionFillDataPointsIdx(Partition p, DM swarm)
{
  PetscInt iel,offset,cellidx,ncells;
  PetscInt *map=NULL;
  PetscInt *point_count,**point_idx;
  PetscFunctionBeginUser;
  /* Get number of cells connected to that partition and their indices */
  ncells = -1; // Initialize to an invalid value
  PetscCall(PartitionGetMap(p,&ncells,&map));
  /* 
    Allocate two arrays to hold the points indices before we generate a unique array 
    This way we avoid allocating very large array before reallocating memory.
  */
  PetscCalloc2(ncells,&point_count,ncells,&point_idx);
  p->ndata = 0; // Initialize the total number of points
  for (iel=0; iel<ncells; iel++) {
    cellidx = map[iel]; // Get the cell index connected to the partition
    /* Find points */
    PetscCall(DMSwarmSortGetPointsPerCell(swarm,cellidx,&point_count[iel],&point_idx[iel]));
    p->ndata += point_count[iel]; // Accumulate the total number of points
  }
  /* Allocate the data indices array */
  PetscCall(PetscMalloc1(p->ndata,&p->data_idx));
  /* Populate */
  offset = 0;
  for (iel=0; iel<ncells; iel++) {
    cellidx = map[iel]; // Get the cell index connected to the partition
    /* Copy the points indices to the data indices array */
    memcpy(&p->data_idx[offset],point_idx[iel],point_count[iel] * sizeof(PetscInt)); 
    offset += point_count[iel]; // Update the offset for the next cell
    PetscCall(DMSwarmSortRestorePointsPerCell(swarm,cellidx,&point_count[iel],&point_idx[iel]));
  }
  PetscFree2(point_count,point_idx); // Free the temporary arrays
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionCreateCoefficientsMat(Partition p, PetscInt nfields)
{
  PetscInt npoints;
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  PetscCheck(!p->X,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Coefficients matrix already created.");
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,nfields,NULL,&p->X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  PetscErrorCode PartitionSetUp(LagRBFCtx ctx, DM swarm, Partition p)
  For the partition:
    - fill the data points indices belonging to the partition
    - create the matrix to hold the solution
    - fill (and factor) the RBF matrix for the partition
*/
PetscErrorCode PartitionSetUp_SinglePartition(LagRBFCtx ctx, DM swarm, Partition p)
{
  PartitionOfUnity pu;
  DMSwarmCellDM    celldm;
  PetscInt         nfields;
  PetscReal        *coords,*values;
  const char       **coordFieldsName;
  PetscFunctionBeginUser;

  if (p->setup) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  /* Swarm */
  PetscCall(DMSwarmSortGetAccess(swarm));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFieldsName));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm, coordFieldsName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm,"fields",&nfields,NULL,(void **)&values));

  /* Setup partition */
  PetscCall(PartitionFillDataPointsIdx(p,swarm));
  PetscCall(PartitionCreateCoefficientsMat(p,nfields));
  PetscCall(pu->psetup(ctx,p,coords));

  /* Restore */
  PetscCall(DMSwarmSortRestoreAccess(swarm));
  PetscCall(DMSwarmRestoreField(swarm,coordFieldsName[0],NULL,NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm,"fields",&nfields,NULL,(void **)&values));

  p->setup = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi)

  Iterates over the points in the interpolation swarm to identify the partitions to which they are connected.
  This avoids setting up partitions that are not needed for the interpolation.
  Needs to be called after each update of the interpolation swarm.
*/
PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi)
{
  PartitionOfUnity pu;
  DMSwarmCellDM    celldm_i;
  Partition        *partitions;
  DM               das;
  PetscInt         np,npoints_i;
  PetscInt         nel,nen;
  PetscInt         *gtol_elidx,*cellid_i;
  const PetscInt   *elidx;
  const char       *cellFieldName_i;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Local DMDA info */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));

  /* Interpolation swarm info */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));

  for (np=0; np<npoints_i; np++) {
    PetscInt k,cidx,local_cidx;
    /* Cell index of the particle */
    cidx       = cellid_i[np];     // parallel DMDA
    local_cidx = gtol_elidx[cidx]; // local sequential DMDA
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      PetscInt  puidx;
      /* Get the partition index */
      puidx = elidx[nen*local_cidx + k];
      p     = partitions[puidx];
      /* Setup the partition (internal check if already setup) */
      PetscCall(PartitionSetUp_SinglePartition(ctx,swd,p));
    }
  }
  PetscCall(DMSwarmRestoreField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMDARestoreElements(das,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBF_PartitionsResetSolved(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  Partition       *partitions;
  PetscInt         npart,n;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  for (n=0; n<npart; n++) {
    partitions[n]->solved = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetMap(Partition p, PetscInt *ncells, PetscInt **map)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (ncells) { *ncells = p->ncells; }
  if (map)    { *map = p->map; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetDataPointsIdx(Partition p, PetscInt *ndata, PetscInt **data_idx)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (ndata)                     { *ndata = p->ndata; }
  if (p->ndata == 0 && data_idx) { *data_idx = NULL; }
  else if (data_idx)             { *data_idx = p->data_idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetCoefficientsMat(Partition p, Mat *X)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (X) { *X = p->X; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetRBFMatrix(Partition p, Mat *A)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (A) { *A = p->A; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetEpsilon(Partition p, PetscReal *epsilon)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Espilon is no longer stored in the Partition structure but in PartitionOfUnity.");
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  //if (epsilon) { *epsilon = p->epsilon; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetEpsilon(Partition p, PetscReal epsilon)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Espilon is no longer stored in the Partition structure but in PartitionOfUnity.");
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  //p->epsilon = epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetIndex(Partition p, PetscInt *idx)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (idx) { *idx = p->idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}