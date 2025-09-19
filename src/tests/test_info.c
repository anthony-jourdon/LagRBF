#include <petsc.h>
#include "lagrangian_rbf.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"
#include "swarm_utils.h"
#include "swarm_interp.h"
#include "swarm_data.h"

static PetscErrorCode ViewUniquePointIdentifier(DM sw)
{
  PetscInt p,npoints;
  PetscInt *pidx;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  for (p=0; p<npoints; p++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Point %d: Unique ID = %" PetscInt_FMT "\n", p, pidx[p]));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionTestGetters(PartitionOfUnity pu, Partition *partition)
{
  PetscInt i,npart;
  PetscFunctionBeginUser;

  PetscCall(PUGetNPartitions(pu,&npart));
  PetscPrintf(PETSC_COMM_WORLD,"Number of partitions: %" PetscInt_FMT "\n",npart);
  
  for (i=0; i<npart; i++) {
    Partition p = partition[i];
    Mat       X;
    PetscInt  k,ncells,ndata;
    PetscInt  *map,*data_idx;
    PetscReal epsilon;
    
    PartitionGetMap(p,&ncells,&map);
    PartitionGetDataPointsIdx(p,&ndata,&data_idx);
    PartitionGetCoefficientsMat(p,&X);
    PartitionGetEpsilon(p,&epsilon);
    PetscPrintf(PETSC_COMM_WORLD,"Partition[%" PetscInt_FMT "]: ncells=%" PetscInt_FMT ", ndata=%" PetscInt_FMT ", epsilon=%g\n",i,ncells,ndata,epsilon);
    if (ncells > 0) {
      for (k=0; k<ncells; k++) {
        PetscPrintf(PETSC_COMM_WORLD,"  map[%" PetscInt_FMT "] = %" PetscInt_FMT "\n",k,map[k]);
      }
    }
    if (ndata > 0) {
      for (k=0; k<ndata; k++) {
        PetscPrintf(PETSC_COMM_WORLD,"  data_idx[%" PetscInt_FMT "] = %" PetscInt_FMT "\n",k,data_idx[k]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataPoints(LagRBFCtx ctx, DM *sw)
{
  DM         swarm;
  PetscInt   dim,nn;
  PetscInt   *m;
  PetscReal  O[3],L[3];
  PetscBool  found;
  char       option_name[PETSC_MAX_PATH_LEN];
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(SwarmDataCreate(ctx,&swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(PetscMalloc1(dim,&m));

  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",prefix));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,m,&nn,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%s not found. User must provide the number of data points in each dimension.",option_name);
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,nn);

  PetscCall(DMGetBoundingBox(ctx->pu->dap,O,L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscCall(PetscFree(m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_interp,swarm_data;
  PetscInt         k;
  PetscReal        overlap,radius,epsilon;
  PetscBool        exists;
  const char       output_dir[] = "output";
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }
  
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Set epsilon parameter */
  overlap = 1.5;
  radius  = overlap / PetscSqrtReal(pu->npart);
  epsilon = 1.0 / radius;
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Set interpolation points */
  PetscCall(SwarmInterpSetUp(ctx,&swarm_interp));
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,&swarm_data));
  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));

  PetscCall(PartitionTestGetters(pu,partitions));
  PetscCall(ViewUniquePointIdentifier(swarm_data));
  
  /* Output */
  PetscCall(PartitionOfUnityMeshView(pu));
  /* output swarm data */
  PetscCall(SwarmOutputCoordsPetscVec(swarm_data,output_dir,NULL));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_data,output_dir,NULL));
  

  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  For inspection, let us limit to 2D cases
  Understanding how 2D is decomposed between processes should be enough
*/

static PetscErrorCode PrintNodesPetscOrdering(DM da, int rank)
{
  PetscInt  sx,sy,lx,ly;
  PetscInt  gsx,gsy,glx,gly;
  PetscInt  i,j;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetCorners(da,&sx,&sy,NULL,&lx,&ly,NULL));
  PetscCall(DMDAGetGhostCorners(da,&gsx,&gsy,NULL,&glx,&gly,NULL));

  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: corner = (%" PetscInt_FMT ", %" PetscInt_FMT "), size = (%" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,sx,sy,lx,ly);
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: ghost corner = (%" PetscInt_FMT ", %" PetscInt_FMT "), ghost size = (%" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,gsx,gsy,glx,gly);
  
  /* Print local indices */
  for (j=0; j<ly; j++) {
    for (i=0; i<lx; i++) {
      PetscInt nidx_local = i + j*lx;
      PetscPrintf(PETSC_COMM_SELF,
        "Rank[%d]: local (%" PetscInt_FMT ", %" PetscInt_FMT "): %" PetscInt_FMT "\n",
        rank,i,j,nidx_local);
    }
  }
  /* Print local ghost indices */
  for (j=0; j<gly; j++) {
    for (i=0; i<glx; i++) {
      PetscInt nidx_local = i + j*glx;
      PetscPrintf(PETSC_COMM_SELF,
        "Rank[%d]: ghost local (%" PetscInt_FMT ", %" PetscInt_FMT "): %" PetscInt_FMT "\n",
        rank,i,j,nidx_local);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintElements(DM da, int rank)
{
  PetscInt       k,iel,nel,nen;
  PetscInt       sex,sey,lex,ley;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  PetscCall(DMDAGetElementsCorners(da,&sex,&sey,NULL));
  PetscCall(DMDAGetElementsSizes(da,&lex,&ley,NULL));
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: element corner = (%" PetscInt_FMT ", %" PetscInt_FMT "), element size = (%" PetscInt_FMT ", %" PetscInt_FMT "), number of elements: %" PetscInt_FMT ", number of nodes: %" PetscInt_FMT "\n",
    rank,sex,sey,lex,ley,nel,nen*nel);

  for (iel=0; iel<nel; iel++) {
    PetscPrintf(PETSC_COMM_SELF,
      "Rank[%d]: Element[%" PetscInt_FMT "]: [",
      rank,iel);
    for (k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];
      PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " ",nidx);
    }
    PetscPrintf(PETSC_COMM_SELF,"]\n");
  }
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintNeighborsRanks(DM da, int rank)
{
  PetscInt          i;
  const PetscMPIInt *neighboring_ranks;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetNeighbors(da, &neighboring_ranks));
  /* 
  In 2D:
    the neighboring_ranks array is of length 9
    the center entry is the rank itself: index 4
  In 3D:
    the neighboring_ranks array is of length 27
    the center entry is the rank itself: index 13
  */

  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: neighboring ranks: [",rank);
  for (i=0; i<9; i++) {
    PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " ",neighboring_ranks[i]);
  }
  PetscPrintf(PETSC_COMM_SELF,"]\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintIS(DM da, int rank)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               n;
  const PetscInt        *idx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(DMGetLocalToGlobalMapping(da,&ltog));
  PetscCall(ISLocalToGlobalMappingGetSize(ltog,&n));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltog,&idx));

  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: local to global mapping: [",rank);
  for (PetscInt i=0; i<n; i++) {
    PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " ",idx[i]);
  }
  PetscPrintf(PETSC_COMM_SELF,"]\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintCoords(DM da, int rank)
{
  DM             cda;
  Vec            coords;
  PetscInt       i,j,sx,sy,sz,nx,ny,nz;
  const PetscReal *LA_coords;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetCorners(da, &sx, &sy, NULL, &nx, &ny, NULL));
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coords));
  PetscCall(VecGetArrayRead(coords,&LA_coords));

  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      PetscInt nidx = i + j*nx;
      PetscPrintf(PETSC_COMM_SELF,
        "Rank[%d]: coords[%" PetscInt_FMT "] (%g, %g)\n",
        rank,nidx,LA_coords[2*nidx],LA_coords[2*nidx+1]);
    }
  }
  PetscCall(VecRestoreArrayRead(coords,&LA_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDACreateRedundantSubDMDA(PartitionOfUnity pu, int rank, DM *sub_da)
{
  IS       is;
  DM       da,cda,seq_da,seq_cda;
  Vec      da_natural_coords,da_coords;
  Vec      seq_coords,seq_natural_coords;
  VecScatter scatter;
  PetscInt M[3],m[3];
  PetscInt sel[3],nel[3];
  PetscInt snodes[3],nnodes[3];
  PetscInt i,j,k,d,idx,npoints,nproc=1;
  PetscInt *natural_gidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: [[ %s ]]\n",rank,PETSC_FUNCTION_NAME);

  /* Get the parallel DMDA */
  PetscCall(PUGetDMDA(pu,&da));

  /* Get the element corners and sizes */
  PetscCall(DMDAGetElementsCorners(da,&sel[0],&sel[1],&sel[2]));
  PetscCall(DMDAGetElementsSizes(da,&nel[0],&nel[1],&nel[2]));
  PetscCall(PetscArrayzero(snodes,3));
  PetscCall(PetscArrayzero(nnodes,3));
  /* Compute the new extent of elements => halo of 1 element around the parallel non-overlapping elements decomposition */
  npoints = 1;
  for (d=0; d<pu->dim; d++) {
    if (sel[d] > 0) sel[d] -= 1;
    if (sel[d] + nel[d] < pu->n[d]-1) nel[d] += 1;
    snodes[d] = sel[d];
    nnodes[d] = nel[d] + 1;
    npoints *= nnodes[d];
  }

  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: new element corner = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), new element size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,sel[0],sel[1],sel[2],nel[0],nel[1],nel[2]);
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: new nodes corner = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), new nodes size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,snodes[0],snodes[1],snodes[2],nnodes[0],nnodes[1],nnodes[2]);

  /* Create a rank-local DMDA of the size of the ghosted parallel DMDA */
  PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nnodes[0],nnodes[1],nproc,nproc,1,1,NULL,NULL,&seq_da));
  PetscCall(DMDASetElementType(seq_da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(seq_da));

  DMDAGetInfo(da,NULL,&M[0],&M[1],&M[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  DMDAGetInfo(seq_da,NULL,&m[0],&m[1],&m[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: parallel DMDA size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,M[0],M[1],M[2]);
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: sequential DMDA size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,m[0],m[1],m[2]);

  /* Create the map of natural global indices between the parallel DMDA and the sequential DMDA */
  PetscMalloc1(pu->dim*npoints,&natural_gidx);
  idx = 0;
  for (j=0; j<nnodes[1]; j++) {
    for (i=0; i<nnodes[0]; i++) {
      /*
      PetscPrintf(PETSC_COMM_SELF,
        "Rank[%d]: (i,j)=(%" PetscInt_FMT ", %" PetscInt_FMT ")\n"
        "         (i+sx,j+sy)=(%" PetscInt_FMT ", %" PetscInt_FMT ")\n"
        "         nidx = %" PetscInt_FMT ", gidx = %" PetscInt_FMT "\n",
        rank,i,j,
        i+snodes[0],j+snodes[1],
        j*nnodes[0] + i, (i+snodes[0]) + (j+snodes[1])*M[0]);
      */
      natural_gidx[2*idx  ] = ((i+snodes[0]) + (j+snodes[1])*M[0])*2;     // x coord
      natural_gidx[2*idx+1] = ((i+snodes[0]) + (j+snodes[1])*M[0])*2 + 1; // y coord
      idx++;
    }
  }
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: npoints: %" PetscInt_FMT ", 2*npoints: %" PetscInt_FMT "\n",
    rank,idx,2*idx);

  /* Get the parallel DMDA global coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&da_coords));
  /* Create a vector to hold coordinates in natural indices */
  PetscCall(DMDACreateNaturalVector(cda,&da_natural_coords));
  /* Fill it */
  PetscCall(DMDAGlobalToNaturalBegin(cda,da_coords,INSERT_VALUES,da_natural_coords));
  PetscCall(DMDAGlobalToNaturalEnd(cda,da_coords,INSERT_VALUES,da_natural_coords));

  /* Create the Index Set to communicate between the parallel DMDA and the sequential DMDA */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,2*npoints,natural_gidx,PETSC_OWN_POINTER,&is));
  
  /* Get the coordinate DM for the sequential DMDA */
  PetscCall(DMGetCoordinateDM(seq_da,&seq_cda));
  /* Create the natural vector for the sequential DMDA */
  PetscCall(DMDACreateNaturalVector(seq_cda,&seq_natural_coords));
  /* 
  Create a global vector that will be used to set coordinates 
  Given that we are on a sequential DMDA, I am not sure that the natural and global ordering are actually different
  */
  PetscCall(DMCreateGlobalVector(seq_cda,&seq_coords));

  /* Scatter the coordinates to the sequential DMDA */
  PetscCall(VecScatterCreate(da_natural_coords,is,seq_natural_coords,NULL,&scatter));
  PetscCall(VecScatterBegin(scatter,da_natural_coords,seq_natural_coords,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,da_natural_coords,seq_natural_coords,INSERT_VALUES,SCATTER_FORWARD));

  /* From natural to global ordering on the sequential DMDA */
  PetscCall(DMDANaturalToGlobalBegin(seq_cda,seq_natural_coords,INSERT_VALUES,seq_coords));
  PetscCall(DMDANaturalToGlobalEnd(seq_cda,seq_natural_coords,INSERT_VALUES,seq_coords));
  /* Set the coordinates */
  PetscCall(DMSetCoordinates(seq_da,seq_coords));

  /* Cleanup */
  PetscCall(VecDestroy(&da_natural_coords));
  PetscCall(VecDestroy(&seq_natural_coords));
  PetscCall(VecDestroy(&seq_coords));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(ISDestroy(&is));

  *sub_da = seq_da;
  PetscFunctionReturn(PETSC_SUCCESS);
}



static PetscErrorCode PartitionCreate_LocalToGlobalPartitionToCellInverseMapping(PartitionOfUnity pu, DM seq_da, Partition *partition)
{
  DM             dap;
  PetscInt       iel,k,nel,nen,nidx;
  const PetscInt *elidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetDMDA(pu,&dap));
  /* Get elements of the parallel DMDA */
  PetscCall(DMDAGetElements(dap,&nel,&nen,&elidx));
  /* Get elements of the sequential DMDA */
  PetscCall(DMDAGetElements(seq_da,&nel,&nen,&elidx));

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

static PetscErrorCode TEST_PartitionCreate(PartitionOfUnity pu, Partition **partition)
{
  Partition *p;
  PetscInt  n,npart;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PetscCalloc1(npart,&p));
  for (n=0; n<npart; n++) { PetscCall(PartitionCreate_SinglePartition(&p[n])); }
  /* Create the inverse map connecting partition to cells 
  TODO: Fix SEGV in parallel
  */
  //PetscCall(PartitionCreate_PartitionToCellInverseMapping(pu,p));
  *partition = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_Parallel()
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               da,seq_da;
  DMDAElementType  etype;
  PetscInt         nel,nen;
  PetscInt         s[3],n[3];
  const PetscInt   *elidx;
  MPI_Comm         comm;
  int              rank,size;
  PetscFunctionBeginUser;
  /* TEST here the decomposition of the domain */
  PetscCall(PartitionOfUnityCreate(&pu));
  PetscCall(PUGetDMDA(pu,&da));

  PetscCall(PetscObjectGetComm((PetscObject)da,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  /* Check element type */
  PetscCall(DMDAGetElementType(da,&etype));
  PetscPrintf(PETSC_COMM_SELF,"Element type: %s\n",etype == DMDA_ELEMENT_P1 ? "P1" : etype == DMDA_ELEMENT_Q1 ? "Q1" : "Unknown");

  PetscCall(DMDACreateRedundantSubDMDA(da,rank,&seq_da));

  /* Elements */
  PetscCall(PrintElements(seq_da,rank));
#if 0
  /* Set the number of local partitions */
  PetscCall(DMDAGetCorners(seq_da,&s[0],&s[1],&s[2],&n[0],&n[1],&n[2]));
  pu->npart = 1; // Initialize
  for (int d=0; d<pu->dim; d++) { pu->npart *= n[d]; }

  PetscCall(TEST_PartitionCreate(pu,&partitions));
#endif

#if 0
  /* Nodes */
  PetscCall(PrintNodesPetscOrdering(da,rank));
  /* Elements */
  PetscCall(PrintElements(da,rank));
  /* Local to global mapping */
  PetscCall(PrintIS(da,rank));
  /* Neighbors */
  PetscCall(PrintNeighborsRanks(da,rank));
  /* Sequential DMDA coords */ 
  PetscCall(PrintCoords(seq_da,rank));
#endif

#if 0
  /* Elements of the parallel DMDA */
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  PetscCall(DMDAGetCorners(seq_da,&seq_sx,&seq_sy,NULL,&seq_lx,&seq_ly,NULL));

  for (int iel=0; iel<nel; iel++) {
    PetscPrintf(PETSC_COMM_SELF,
      "Rank[%d]: Element[%" PetscInt_FMT "] (",rank,iel);
    for (int k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];
      PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " ",nidx);
    }
    PetscPrintf(PETSC_COMM_SELF,")\n");
  }
  for (int j=0; j<seq_ly; j++) {
    for (int i=0; i<seq_lx; i++) {
      PetscInt nidx_local = i + j*seq_lx;
      PetscPrintf(PETSC_COMM_SELF,
        "Rank[%d]: seq local (%" PetscInt_FMT ", %" PetscInt_FMT "): %" PetscInt_FMT "\n",
        rank,i,j,nidx_local);
    }
  }
#endif
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));


  PetscCall(PartitionOfUnityDestroy(&pu));
  PetscCall(DMDestroy(&seq_da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ViewSeqDMDA(DM da, int rank)
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  Vec               testvec;
  char              prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"da_seq_%d_",rank);

  PetscCall(PetscOptionsCreateViewer(PETSC_COMM_SELF,NULL,prefix,"-view",&viewer,&format,NULL));
  /* Create a vec just to print out the mesh */
  PetscCall(DMCreateGlobalVector(da,&testvec));
  PetscCall(VecSet(testvec,1.0));
  PetscCall(VecView(testvec,viewer));
  PetscCall(VecDestroy(&testvec));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_Parallel_2()
{
  PartitionOfUnity pu;
  DM               da,dae,seq_da;
  PetscInt         stencil_width=1;
  PetscInt         snodes[3],lnodes[3],sghost[3],lghost[3],sel[3],nel[3];
  PetscInt         seq_snodes[3],seq_lnodes[3],seq_sel[3],seq_nel[3];
  PetscBool        exists;
  MPI_Comm         comm;
  int              rank,size;
  const char       output_dir[] = "output_sw2";
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }

  PetscCall(PartitionOfUnityCreate(&pu));
  PetscCall(PUGetDMDA(pu,&da));

  PetscCall(PetscObjectGetComm((PetscObject)da,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  PetscCall(DMDAGetCorners(da,&snodes[0],&snodes[1],&snodes[2],&lnodes[0],&lnodes[1],&lnodes[2]));
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: Nodes corner = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,snodes[0],snodes[1],snodes[2],lnodes[0],lnodes[1],lnodes[2]);
  PetscCall(DMDAGetGhostCorners(da,&sghost[0],&sghost[1],&sghost[2],&lghost[0],&lghost[1],&lghost[2]));
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: Nodes ghost corner = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,sghost[0],sghost[1],sghost[2],lghost[0],lghost[1],lghost[2]);

  PetscCall(DMDAGetElementsCorners(da,&sel[0],&sel[1],&sel[2]));
  PetscCall(DMDAGetElementsSizes(da,&nel[0],&nel[1],&nel[2]));
  PetscPrintf(PETSC_COMM_SELF,
    "Rank[%d]: Elements corner = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), size = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n",
    rank,sel[0],sel[1],sel[2],nel[0],nel[1],nel[2]);

  PetscCall(DMDACreateRedundantSubDMDA(pu,rank,&seq_da));

  PetscCall(PrintCoords(seq_da,rank));
  PetscCall(ViewSeqDMDA(seq_da, rank));
  

  PetscCall(PartitionOfUnityDestroy(&pu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  //PetscCall(Test());
  //PetscCall(Test_Parallel());
  PetscCall(Test_Parallel_2());

  PetscCall(PetscFinalize());
  
  return 0;
}