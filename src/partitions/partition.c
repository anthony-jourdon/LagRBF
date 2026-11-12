/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    partition.c

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "kernels/kernels_rbf.h"

/*@C
  PartitionCreate_PartitionToCellInverseMapping - Create the inverse mapping from partition to cells.

  Collective

  Input Parameters:
+ pu (PartitionOfUnity) - the PartitionOfUnity context
. partition (Partition*) - array of Partition structures

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  Create the inverse map from partition to cells.
  The partition is a node of the DMDA, and the cells are the elements of the DMDA.
  The map is an array of indices, where each index corresponds to a cell that is connected to the partition.
  The number of cells connected to each partition is stored in ``ncells``.
  A map array is allocated for each partition and filled with the cell indices.
  Memory is freed in :c:func:`PartitionDestroy_SinglePartition()` for a given partition or in :c:func:`PartitionDestroy()` for all partitions.

.seealso: PartitionCreate()
@*/
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

/*@C
  PartitionCreate_SinglePartition - Create a single partition structure.

  Collective

  Input Parameters:
. partition (Partition*) - pointer to the Partition structure to create

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function allocates memory for a single partition structure and initializes its members to default values.
  Memory is freed in :c:func:`PartitionDestroy_SinglePartition()` for a given partition.

.seealso: PartitionCreate(), PartitionDestroy_SinglePartition(), PartitionDestroy()
@*/
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

/*@C
  PartitionDestroy_SinglePartition - Destroy a single partition structure.

  Collective

  Input Parameters:
. partition (Partition*) - pointer to the Partition structure to destroy
. destroy (PetscErrorCode (*)(Partition)) - function to destroy the partition specific data (can be NULL if not needed)

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function frees the memory allocated for a single partition structure.
  It is called by :c:func:`PartitionDestroy()` for each partition.

.seealso: PartitionDestroy(), PartitionCreate_SinglePartition(), PartitionCreate()
@*/
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

static PetscErrorCode PartitionReset_SinglePartition(Partition p, PetscErrorCode (*destroy)(Partition))
{
  PetscFunctionBeginUser;
  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  p->solved = PETSC_FALSE;
  p->setup  = PETSC_FALSE;
  if (p->data_idx) { 
    PetscCall(PetscFree(p->data_idx)); 
    p->data_idx = NULL;
    p->ndata    = 0;
  }
  if (p->X)    { PetscCall(MatDestroy(&p->X));      }
  if (p->A)    { PetscCall(MatDestroy(&p->A));      }
  if (destroy) { PetscCall(destroy(p));             }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PartitionCreate - Create and initialize all partitions.

  Collective

  Input Parameters:
+ pu (PartitionOfUnity) - the PartitionOfUnity context
. partition (Partition*) - array of Partition structures

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function creates and initializes all partitions of the domain.
  It allocates memory for each partition and sets up the inverse mapping from partitions to cells.
  Memory is freed in :c:func:`PartitionDestroy()` for all partitions.

.seealso: PartitionDestroy(), PartitionCreate_SinglePartition(), PartitionDestroy_SinglePartition(), PartitionCreate_PartitionToCellInverseMapping()
@*/
PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition)
{
  Partition *p;
  PetscInt  n,npart;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscCall(PetscCalloc1(npart,&p));
  for (n=0; n<npart; n++) { PetscCall(PartitionCreate_SinglePartition(&p[n])); }
  /* Create the inverse map connecting partition to cells */
  PetscCall(PartitionCreate_PartitionToCellInverseMapping(pu,p));
  /* Set the index of each partition */
  PetscCall(PartitionsSetIndex(pu,p));

  *partition = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PartitionDestroy - Destroy all partitions in the array.

  Collective

  Input Parameters:
+ pu (PartitionOfUnity) - the PartitionOfUnity context
. partition (Partition*) - array of Partition structures

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function destroys all partitions in the array by freeing their associated data.

.seealso: PartitionCreate(), PartitionDestroy_SinglePartition()
@*/
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

/*@C
  PartitionFillDataPointsIdx - Fill the data points indices for a partition.

  Collective

  Input Parameters:
+ p (Partition) - the Partition data structure
. swarm (DM)    - the swarm data structure

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function fills the data points indices for a partition by retrieving the points associated with each cell connected to the partition.
@*/
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

static PetscErrorCode PartitionSetUp_SinglePartition(LagRBFCtx ctx, Partition p, DM swarm, PetscInt nfields, PetscReal *coords)
{
  PartitionOfUnity pu;
  PetscFunctionBeginUser;
  if (p->setup) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PartitionFillDataPointsIdx(p,swarm));
  PetscCall(PartitionCreateCoefficientsMat(p,nfields));
  PetscCall(pu->psetup(ctx,p,coords));
  p->setup = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  LagRBF_PartitionsSetUp - Set up all partitions and solve the coefficients for interpolation.

  Collective

  Input Parameters:
+ ctx (LagRBFCtx) - the LagRBF context
. swd (DM)        - the data swarm DM
. swi (DM)        - the interpolation swarm DM

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function sets up all partitions and solves for the coefficients needed for interpolation.
  It retrieves the necessary information from the PartitionOfUnity context and the swarms, and iterates over each point in the interpolation swarm to set up and solve the corresponding partitions.
  Partitions are set up and solved only once, even if multiple points are connected to the same partition. Thus the necessity to reset them with :c:func:`PartitionReset()` between calls to this function if the data swarm changes.

.seealso: PartitionReset()
@*/
PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi)
{
  PartitionOfUnity pu;
  DMSwarmCellDM    celldm_i,celldm_d;
  Partition        *partitions;
  DM               das;
  PetscInt         np,nfields,nel,nen,npoints_i;
  PetscInt         *gtol_elidx,*cellid_i;
  const PetscInt   *elidx;
  PetscReal        *coords_d,*values_d;
  const char       **coordFieldsName_d,*cellFieldName_i;
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

  /* Data swarm info */
  PetscCall(DMSwarmSortGetAccess(swd));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd, &celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d, NULL, &coordFieldsName_d));
  /* Get fields */
  PetscCall(DMSwarmGetField(swd, coordFieldsName_d[0], NULL, NULL, (void **)&coords_d));
  PetscCall(DMSwarmGetField(swd,"fields",&nfields,NULL,(void **)&values_d));

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
      PetscCall(PartitionSetUp_SinglePartition(ctx,p,swd,nfields,coords_d));
      /* Solve for coefficients (internal check if already solved for) */
      PetscCall(PartitionSolveInterpolationCoefficients_SinglePartition(ctx,p,nfields,values_d));
    }
  }  
  /* Restore */
  PetscCall(DMSwarmSortRestoreAccess(swd));
  PetscCall(DMSwarmRestoreField(swd,coordFieldsName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMSwarmRestoreField(swd,"fields",&nfields,NULL,(void **)&values_d));
  PetscCall(DMSwarmRestoreField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMDARestoreElements(das,&nel,&nen,&elidx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  LagRBF_PartitionsReset - Reset the partitions to their initial state, clearing any computed coefficients and data point indices.

  Collective

  Input Parameters:
+ ctx (LagRBFCtx) - the LagRBF context

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function resets the partitions based on the solve method specified in the LagRBFCtx context. It clears any computed coefficients and data point indices, preparing the partitions for a new setup or solve.
@*/
PetscErrorCode LagRBF_PartitionsReset(LagRBFCtx ctx)
{
  PartitionOfUnity pu;
  Partition       *partitions;
  PetscInt         npart,n;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PUGetNPartitionsLocal(pu,&npart));

  switch (ctx->solve_method) {
    case A_FACTOR:
    case A_KSP:
      for (n=0; n<npart; n++) {
        partitions[n]->solved = PETSC_FALSE;
      }
      break;

    case A_KSP_ASMB:
      for (n=0; n<npart; n++) {
        PetscCall(PartitionReset_SinglePartition(partitions[n],pu->pdestroy));
      }
      break;
    
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown solve method %" PetscInt_FMT,ctx->solve_method);
      break;
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

PetscErrorCode PartitionGetShapeFactor(Partition p, PetscReal *epsilon)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Shape Factor is no longer stored in the Partition structure but in PartitionOfUnity.");
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  //if (epsilon) { *epsilon = p->epsilon_r; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSetShapeFactor(Partition p, PetscReal epsilon)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Shape Factor is no longer stored in the Partition structure but in PartitionOfUnity.");
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  //p->epsilon_r = epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionGetIndex(Partition p, PetscInt *idx)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  if (idx) { *idx = p->idx; }
  PetscFunctionReturn(PETSC_SUCCESS);
}