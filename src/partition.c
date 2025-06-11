#include "partition_of_unity.h"
#include "partition.h"

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
  DM             dap;
  PetscInt       iel,k,nel,nen,nidx;
  const PetscInt *elidx;
  PetscFunctionBeginUser;

  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(DMDAGetElements(dap,&nel,&nen,&elidx));

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
  PetscCall(PUGetDMDA(pu,&da));
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
  VecCreate(PETSC_COMM_SELF,&p->X); // Vector holding coefficients for the partition interpolation
  p->A        = NULL; // RBF matrix for the partition
  p->ncells   = 0; // Initialize number of cells connected to the partition
  p->map      = NULL; // Initialize the map
  p->ndata    = 0; // Initialize number of data points in the partition
  p->data_idx = NULL; // Initialize the data points indices
  p->epsilon  = 1.0; // Default epsilon for the RBF kernel
  p->idx      = -1; // Index of the partition
  
  *partition = p;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionDestroy_SinglePartition(Partition *partition)
{
  Partition p = *partition;
  PetscFunctionBeginUser;

  if (!p)          { PetscFunctionReturn(PETSC_SUCCESS); }
  if (p->map)      { PetscCall(PetscFree(p->map));       }
  if (p->data_idx) { PetscCall(PetscFree(p->data_idx));  }
  if (p->X)        { PetscCall(VecDestroy(&p->X));       }
  if (p->A)        { PetscCall(MatDestroy(&p->A));       }
  PetscCall(PetscFree(p));
  *partition = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition)
{
  Partition *p;
  PetscInt  n,npart;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PetscCalloc1(npart,&p));
  for (n=0; n<npart; n++) { PetscCall(PartitionCreate_SinglePartition(&p[n])); }
  /* Create the inverse map connecting partition to cells */
  PetscCall(PartitionCreate_PartitionToCellInverseMapping(pu,p));
  *partition = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition)
{
  PetscInt n,npart;
  Partition *p = *partition;
  PetscFunctionBeginUser;

  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(PUGetNPartitions(pu,&npart));
  for (n=0; n<npart; n++) { PetscCall(PartitionDestroy_SinglePartition(&p[n])); }
  PetscCall(PetscFree(p));
  *partition = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionFillDataPointsIdx(Partition p, DM swarm)
{
  PetscInt      iel,ncells,offset,cellidx;
  PetscInt     *map,*point_count,**point_idx;
  PetscFunctionBeginUser;
  /* Get number of cells connected to that partition and their indices */
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

PetscErrorCode PartitionFillDataPointsIdx_All(Partition *partitions, PetscInt npart, DM swarm)
{
  PetscInt n;
  PetscFunctionBeginUser;
  PetscCall(DMSwarmSortGetAccess(swarm));
  for (n=0; n<npart; n++) { PetscCall(PartitionFillDataPointsIdx(partitions[n],swarm)); }
  PetscCall(DMSwarmSortRestoreAccess(swarm));
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

PetscErrorCode PartitionGetCoefficientsVec(Partition p, Vec *X)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (X) { *X = p->X; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetEpsilon(Partition p, PetscReal *epsilon)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (epsilon) { *epsilon = p->epsilon; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetEpsilon(Partition p, PetscReal epsilon)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  p->epsilon = epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetIndex(Partition p, PetscInt *idx)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (idx) { *idx = p->idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}