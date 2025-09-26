#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"

PetscErrorCode PUCreate(PartitionOfUnity *pu)
{
  PartitionOfUnity p;
  MPI_Comm         comm = PETSC_COMM_WORLD;
  int              rank;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscNew(&p));
  PetscCall(PetscSNPrintf(p->name, PETSC_MAX_PATH_LEN-1,"pu_mesh"));
  PetscCall(PetscSNPrintf(p->sname, PETSC_MAX_PATH_LEN-1,"pu_mesh_%d",rank));
  p->dap   = NULL;
  p->das   = NULL;
  p->dim   = 0;
  p->n     = NULL;
  p->npart = 0;
  p->ptos  = NULL;
  p->O     = NULL;
  p->L     = NULL;

  *pu = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUSetFromOptions(PartitionOfUnity pu)
{
  PetscInt  nn,d;
  char      prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscBool found;
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PUCreate() first.");
  /* Set prefix for options */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN-1, "%s_", pu->name));

  /* Spatial dimension */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sdim", prefix));
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&pu->dim,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not provided. Use it to set the spatial dimension of the problem.",option_name);
  PetscCheck(pu->dim >= 1 && pu->dim <= 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Spatial dimension must be between 1 and 3. Found: %" PetscInt_FMT,pu->dim);
  
  /* Number of partition in each dimension */
  PetscCall(PetscMalloc1(pu->dim,&pu->n));
  nn = pu->dim; // Number of entries expected
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sn", prefix));
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,pu->n,&nn,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not provided. Use it to set the number of partitions in each spatial dimension.",option_name);
  PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn);
  
  /* Bounding box */
  /* Origin */
  PetscCall(PetscMalloc1(pu->dim,&pu->O));
  for (d=0; d<pu->dim; d++) { pu->O[d] = 0.0; } // Default origin
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sO",prefix));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,option_name,pu->O,&nn,&found));
  if (found) { PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn); }
  /* Length */
  PetscCall(PetscMalloc1(pu->dim,&pu->L));
  for (d=0; d<pu->dim; d++) { pu->L[d] = 1.0; } // Default length
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sL",prefix));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,option_name,pu->L,&nn,&found));
  if (found) { PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PULoadFromFile(PartitionOfUnity pu, const char filename[])
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUCreateDMDA(PartitionOfUnity pu)
{
  DM       da;
  PetscInt stencil_width=1;
  char     prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  switch (pu->dim) {
    case 1:
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,pu->n[0],1,stencil_width,NULL,&da));
    break;

    case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,pu->n[0],pu->n[1],PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,&da));
    break;

    case 3: 
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,pu->n[0],pu->n[1],pu->n[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,NULL,&da));
    break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Dimension can only be 1, 2 or 3. Found: %" PetscInt_FMT,pu->dim);
  }
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(PetscObjectSetName((PetscObject)da,pu->name));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN-1, "%s_",pu->name));
  PetscCall(DMSetOptionsPrefix(da,prefix));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,pu->O[0],pu->L[0],pu->O[1],pu->L[1],pu->O[2],pu->L[2]));
  
  pu->dap = da;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUCreateDMDALocal(PartitionOfUnity pu)
{
  VecScatter scatter;
  IS         is;
  DM         dap=NULL;
  DM         das,cdap,cdas;
  Vec        dap_coor,dap_nat_coor;
  Vec        das_coor;
  PetscInt   *natural_gidx;
  PetscInt   M[3],start_el[3],size_el[3];
  PetscInt   par_start_el[] = {0,0,0};
  PetscInt   par_size_el[]  = {1,1,1};
  PetscInt   start_node[] = {0,0,0};
  PetscInt   size_node[]  = {1,1,1};
  PetscInt   local_start_el[] = {0,0,0};
  PetscInt   i,j,k,d,dim,npoints,stencil_width,cnt,par_nel;
  PetscFunctionBeginUser;
  /* Get parallel DMDA */
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(DMDAGetInfo(dap,&dim,&M[0],&M[1],&M[2],NULL,NULL,NULL,NULL,&stencil_width,NULL,NULL,NULL,NULL));
  /* Get the (rank-local) element corners and sizes */
  PetscCall(DMDAGetElementsCorners(dap,&start_el[0],&start_el[1],&start_el[2]));
  PetscCall(DMDAGetElementsSizes(dap,&size_el[0],&size_el[1],&size_el[2]));
  for (d=0; d<dim; d++) {
    par_start_el[d] = start_el[d];
    par_size_el[d]  = size_el[d];
  }
  /* Compute the new extent of elements => halo of 1 element around the parallel non-overlapping elements decomposition */
  npoints = 1;
  par_nel = 1;
  for (d=0; d<dim; d++) {
    /* If not on the boundary, expand the element indices */
    if (start_el[d] > 0)                   { start_el[d] -= 1; } // Not on the lower boundary, expand lower bound
    if (start_el[d] + size_el[d] < M[d]-1) { size_el[d]  += 1; } // Not on the upper boundary after expanding the start, expand size
    /* Add one more if after expansion we are still not on the boundary ==> subdomain surrounded on both sides in d direction */
    if (start_el[d] > 0 && start_el[d] + size_el[d] < M[d]-1) { size_el[d]  += 1; } // Not on the lower and upper boundary, expand size again
    /* Starting element of the parallel DMDA in the sequential DMDA */
    local_start_el[d] = par_start_el[d] - start_el[d];
    /* Convert in node extent */
    start_node[d] = start_el[d];
    size_node[d]  = size_el[d] + 1;
    /* Count how much nodes in total */
    npoints      *= size_node[d];
    /* Count how much elements in the parallel non-overlapping mesh */
    par_nel      *= par_size_el[d];
  }
  /* Set the number of partitions on that rank */
  pu->npart = npoints;
  /* Create the map between the parallel and sequential DMDA elements */
  PetscCall(PetscMalloc1(par_nel,&pu->ptos));
  for (k=0; k<par_size_el[2]; k++) {
    for (j=0; j<par_size_el[1]; j++) {
      for (i=0; i<par_size_el[0]; i++) {
        PetscInt par_eidx,seq_eidx;
        par_eidx = i + j*par_size_el[0] + k*par_size_el[0]*par_size_el[1];
        seq_eidx = (i + local_start_el[0]) + (j + local_start_el[1])*size_el[0] + (k + local_start_el[2])*size_el[0]*size_el[1];
        pu->ptos[par_eidx] = seq_eidx;
      }
    }
  }

  /* Create a new sequential DMDA of the halo size, each rank owns its own DMDA */
  switch (dim) {
    case 1:
      PetscCall(DMDACreate1d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,size_node[0],1,stencil_width,NULL,&das));
      break;

    case 2:
      PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,size_node[0],size_node[1],PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,&das));
      break;

    case 3:
      PetscCall(DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,size_node[0],size_node[1],size_node[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,NULL,&das));
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Dimension can only be 1, 2 or 3. Found: %" PetscInt_FMT,dim);
  }
  PetscCall(DMDASetElementType(das, DMDA_ELEMENT_Q1));
  PetscCall(PetscObjectSetName((PetscObject)das,pu->sname));
  PetscCall(DMSetUp(das));

  /* Create the map of natural global indices between the parallel DMDA and the sequential DMDA */
  PetscMalloc1(dim*npoints,&natural_gidx);
  cnt = 0;
  for (k=0; k<size_node[2]; k++) {
    for (j=0; j<size_node[1]; j++) {
      for (i=0; i<size_node[0]; i++) {
        /* Compute the global index in natural order */
        PetscInt gidx = (i+start_node[0]) + (j+start_node[1])*M[0] + (k+start_node[2])*M[0]*M[1];
        for (d=0; d<dim; d++) {
          natural_gidx[dim*cnt + d] = gidx*dim + d; // Each node has 'dim' coordinates
        }
        cnt++;
      }
    }
  }
  /* Create the Index Set to communicate between the parallel DMDA and the sequential DMDA */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,dim*npoints,natural_gidx,PETSC_OWN_POINTER,&is));

  /* Get the parallel DMDA coordinates */
  PetscCall(DMGetCoordinateDM(dap,&cdap));
  PetscCall(DMGetCoordinates(dap,&dap_coor));
  /* Create a vector to hold the coordinates in natural order */
  PetscCall(DMDACreateNaturalVector(cdap,&dap_nat_coor));
  /* Fill it */
  PetscCall(DMDAGlobalToNaturalBegin(cdap,dap_coor,INSERT_VALUES,dap_nat_coor));
  PetscCall(DMDAGlobalToNaturalEnd(cdap,dap_coor,INSERT_VALUES,dap_nat_coor));
  
  /* Get the coordinate DM for the sequential DMDA */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  /* Create the vector to hold the coordinates for the sequential DMDA, sequential so natural = global */
  PetscCall(DMCreateGlobalVector(cdas,&das_coor));

  /* Scatter the coordinates to the sequential DMDA */
  PetscCall(VecScatterCreate(dap_nat_coor,is,das_coor,NULL,&scatter));
  PetscCall(VecScatterBegin(scatter,dap_nat_coor,das_coor,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,dap_nat_coor,das_coor,INSERT_VALUES,SCATTER_FORWARD));

  /* Set the coordinates */
  PetscCall(DMSetCoordinates(das, das_coor));

  /* Clean up */
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(ISDestroy(&is));
  PetscCall(VecDestroy(&das_coor));
  PetscCall(VecDestroy(&dap_nat_coor));

  pu->das = das;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionOfUnityCreate(PartitionOfUnity *pu)
{
  PartitionOfUnity p;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUCreate(&p));
  PetscCall(PUSetFromOptions(p));
  PetscCall(PUCreateDMDA(p));
  /* Create the local DMDA to hold partitions */
  PetscCall(PUCreateDMDALocal(p));

  *pu = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionOfUnityDestroy(PartitionOfUnity *pu)
{
  PartitionOfUnity p = *pu;
  PetscFunctionBeginUser;

  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (p->dap)  { PetscCall(DMDestroy(&p->dap)); }
  if (p->das)  { PetscCall(DMDestroy(&p->das)); }
  if (p->n)    { PetscCall(PetscFree(p->n)); }
  if (p->O)    { PetscCall(PetscFree(p->O)); }
  if (p->L)    { PetscCall(PetscFree(p->L)); }
  if (p->ptos) { PetscCall(PetscFree(p->ptos)); }
  PetscCall(PetscFree(p));
  *pu = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetDMDA(PartitionOfUnity pu, DM *dap)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  PetscCheck(pu->dap,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Partition of Unity mesh not created yet. Call PUCreateDMDA() first.");
  if (dap) { *dap = pu->dap; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetDMDALocal(PartitionOfUnity pu, DM *das)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  PetscCheck(pu->das,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Partition of Unity local mesh not created yet. Call PUCreateDMDALocal() first.");
  if (das) { *das = pu->das; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetNPartitions(PartitionOfUnity pu, PetscInt *npart)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  if (npart) { *npart = pu->npart; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetDimension(PartitionOfUnity pu, PetscInt *dim)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  if (dim) { *dim = pu->dim; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetGlobalToLocalElementMap(PartitionOfUnity pu, PetscInt **ptos)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  PetscCheck(pu->ptos,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Partition of Unity parallel to sequential element map not created yet.");
  if (ptos) { *ptos = pu->ptos; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  View the Mesh associated with the Partition Of Unity DMDA.
  Mostly for debugging purposes.
  Use options to control the output.
  Example usage for a vts output: 
    -pu_mesh_view vtk:pu_mesh.vts:VTK_VTS:write
  See PetscOptionsCreateViewer() manual page for more details.
*/
static PetscErrorCode PartitionOfUnity_DMDAView(DM da)
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  Vec               testvec;
  const char        *dm_name;
  char              prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(PetscObjectGetName((PetscObject)da,&dm_name));
  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"%s_",dm_name));
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)da),NULL,prefix,"-view",&viewer,&format,NULL));
  /* Create a vec just to print out the mesh */
  PetscCall(DMCreateGlobalVector(da,&testvec));
  PetscCall(VecSet(testvec,1.0));
  PetscCall(VecView(testvec,viewer));
  PetscCall(VecDestroy(&testvec));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Output the parallel DMDA */
PetscErrorCode PartitionOfUnityMeshView(PartitionOfUnity pu)
{
  DM dap=NULL;
  PetscFunctionBeginUser;
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PartitionOfUnity_DMDAView(dap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Output sequential DMDAs */
PetscErrorCode PartitionOfUnityMeshView_Seq(PartitionOfUnity pu)
{
  DM         das=NULL;
  PetscFunctionBeginUser;
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(PartitionOfUnity_DMDAView(das));
  PetscFunctionReturn(PETSC_SUCCESS);
}