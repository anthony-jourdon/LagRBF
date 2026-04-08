#include <petsc.h>
#include "lagrangian_rbf.h"
#include "kernels/kernels_rbf.h"
#include "partitions/partition_solve.h"
#include "partitions/partition.h"
#include "io_utils.h"
#include "partition_of_unity.h"
#include "cell_info.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_output.h"
#include "readers/stagYY/binary_loader.h"

#if 0
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

  PetscCall(PUGetNPartitionsLocal(pu,&npart));
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
  const char       *output_data[] = {"fields"};
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
  PetscCall(SwarmOutput(ctx,swarm_data,1,output_data,PETSC_FALSE));

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

static PetscErrorCode Test_3()
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               dap,das;
  PetscInt         *ptos;
  const PetscInt   *elidx,*elidx_s;
  PetscInt         nel,nen;
  PetscBool        exists;
  MPI_Comm         comm;
  int              rank,size;
  const char       output_dir[] = "test_elmap";
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }

  PetscCall(PartitionOfUnityCreate(&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PUGetDMDALocal(pu,&das));

  PetscCall(PetscObjectGetComm((PetscObject)dap,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  PetscCall(PartitionCreate(pu,&partitions));

  for (int n=0; n<pu->npart; n++) {
    Partition p = partitions[n];
    PetscInt ncells;
    PetscInt *map;
    PartitionGetMap(p,&ncells,&map);
    PetscPrintf(PETSC_COMM_SELF,
      "Rank[%d]: Partition[%" PetscInt_FMT "]: ncells=%" PetscInt_FMT "\n",
      rank,n,ncells);
    for (int k=0; k<ncells; k++) {
      PetscPrintf(PETSC_COMM_SELF,"  Rank[%d]: map[%" PetscInt_FMT "] = %" PetscInt_FMT "\n",rank,k,map[k]);
    }
  }
  
  PetscCall(PUGetGlobalToLocalElementMap(pu,&ptos));
  PetscCall(DMDAGetElements(dap,&nel,&nen,&elidx));
  for (int i=0; i<nel; i++) {
    PetscPrintf(PETSC_COMM_SELF,
      "Rank[%d] parallel el[%d] = seq el[%" PetscInt_FMT "]\n",
      rank,i,ptos[i]);
  }

  //PetscCall(ViewSeqDMDA(das, rank));
  PetscCall(PartitionOfUnityMeshView_Seq(pu));
  PetscCall(PartitionOfUnityMeshView(pu));
  PetscCall(PartitionDestroy(pu,&partitions));
  PetscCall(PartitionOfUnityDestroy(&pu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_4()
{
  DMSwarmCellDM    celldm;
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  CellInfo         *ci;
  DM               das,swarm_data;
  PetscInt         iel,nel,nen;
  PetscInt         *cellid;
  const PetscInt   *elidx;
  PetscReal        *coords;
  int              rank;
  const char       **coordFieldName;
  const char       *cellFieldName;
  const char       *data_output[] = {"fields"};
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));
  /* Create data points */
  PetscCall(CreateDataPoints(ctx,&ctx->swd));
  /* Get structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));
  PetscCall(LagRBFGetSwarmData(ctx,&swarm_data));
  PetscCall(SwarmOutput(ctx, swarm_data, 1, data_output, PETSC_FALSE));
  PetscCall(PartitionOfUnityMeshView(pu));
  /* Get local sequential DMDA */
  PetscCall(PUGetDMDALocal(pu,&das));
  /* Get Elements info */
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  /* Setup cell info */
  PetscCall(CellInfoSetUp(pu,ci,swarm_data));

  /* Print cell info content */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm,NULL,&coordFieldName));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellFieldName));
  PetscCall(DMSwarmGetField(swarm_data, coordFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm_data, cellFieldName, NULL, NULL, (void **)&cellid));
  for (iel=0; iel<nel; iel++) {
    CellInfo c = ci[iel];
    PetscInt nd,eidx,nneigh,ndata;
    PetscInt *neighbours,*data_idx;

    PetscCall(CellInfoGetIndex(c,&eidx));
    PetscCall(CellInfoGetNeighbours(c,&nneigh,&neighbours));
    PetscCall(CellInfoGetDataPointsIdx(c,&ndata,&data_idx));
    PetscPrintf(PETSC_COMM_SELF,
      "Rank[%d]: CellInfo[%" PetscInt_FMT "]: eidx = %" PetscInt_FMT ", nneigh = %" PetscInt_FMT ", ndata = %" PetscInt_FMT "\n",
      rank,iel,eidx,nneigh,ndata);
    for (nd=0; nd<ndata; nd++) {
      PetscInt pidx = data_idx[nd];
      PetscPrintf(
        PETSC_COMM_SELF,
        "  Rank[%d]: data_idx[%" PetscInt_FMT "] = %" PetscInt_FMT ", cellidx = %" PetscInt_FMT ", coords = (%1.6e, %1.6e)\n",
        rank,nd,data_idx[nd],cellid[pidx],coords[2*pidx],coords[2*pidx+1]
      );
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_data, coordFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm_data, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMDARestoreElements(das,&nel,&nen,&elidx));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
PetscErrorCode TestMemoryUsage()
{
  PartitionOfUnity pu;
  LagRBFCtx ctx;
  Partition *partitions;
  CellInfo  *cellinfo;
  DM        swarm_data,swarm_interp;
  PetscInt  np,cnt,npoints_i,npoints_d,npart;
  PetscInt  *rm_idx;
  PetscBool exists;
  PetscBool scalars[] = {PETSC_FALSE};
  PetscMPIInt rank;
  //const char *output_fields[] = { DMSwarmPICField_coor, "fields" };
  char *fname[] = { "/mnt/c/Users/jourdon/Documents/Scripts/PostProc/StagYY/PJB6_YS1_Rh32_vp00039" };
  const char *output_fields[] = {"DMSwarm_pid","fields"};
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&cellinfo));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Read data only on rank 0 */
  if (rank == 0) {
    PetscCall(StagBinSwarmSetFromFiles(swarm_data, fname, 1, scalars));
  }
  /* Send data to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx,swarm_data));
  /* Duplicate ghost points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,swarm_data));
  {
    DMSwarmCellDM    celldm;
    PetscInt         dim,n;
    const char       **coordFieldsName;
    PetscReal        *coords;
    PetscCall(DMGetDimension(swarm_data,&dim));
    PetscCall(DMSwarmSortGetAccess(swarm_data));
    PetscCall(PUGetNPartitionsLocal(pu,&npart));
    /* Setup for coords */
    PetscCall(DMSwarmGetCellDMActive(swarm_data, &celldm));
    PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFieldsName));
    /* Get fields */
    PetscCall(DMSwarmGetField(swarm_data, coordFieldsName[0], NULL, NULL, (void **)&coords));
    for (n=0; n<npart; n++) {
      Partition p = partitions[n];
      PetscInt  npoints,i,j,d,zero_count;
      PetscInt  *pidx;
      PetscReal epsilon;
      PetscCall(PartitionFillDataPointsIdx(p,swarm_data));
      PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
      if (npoints == 0) continue;
      PetscCall(PartitionGetEpsilon(p,&epsilon));
      /* RBF matrix */
      zero_count = 0;
      for (i=0; i<npoints; i++) {
        for (j=i; j<npoints; j++) {
          PetscReal dist = 0.0;
          /* Compute the distance between points */
          for (d=0; d<dim; d++) {
            dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
          }
          dist = PetscSqrtReal(dist);
          /* Fill the upper triangular part with the RBF kernel */
          if (1.0 - epsilon * dist < 0.0) {
            zero_count++;
          }
        }
      }
      PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Partition[%" PetscInt_FMT "] has %" PetscInt_FMT " data points and 2*%" PetscInt_FMT " zero entries in the RBF matrix\n",rank,n,npoints,zero_count);
    }
    PetscCall(DMSwarmSortRestoreAccess(swarm_data));
  }

  PetscCall(PartitionOfUnityMeshView(pu));
  PetscCall(PartitionOfUnityMeshView_Seq(pu));

  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Test_CellInfo()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *cellinfo;
  DM               das;
  PetscInt         nel,nen,iel,sw,dim;
  const PetscInt   *elidx;
  PetscMPIInt      rank;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&cellinfo));

  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  PetscCall(DMGetDimension(das,&dim));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test_Load()
{
  LagRBFCtx        ctx;
  DM               swarm_interp;
  PetscMPIInt      rank;
  PetscBool        exists,found;
  char             restart_dir[PETSC_MAX_PATH_LEN];
  const char       *output_fields[] = {"DMSwarm_pid"};
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-restart_directory",restart_dir,PETSC_MAX_PATH_LEN-1,&found));
  PetscCall(PetscTestDirectory(restart_dir,'r',&exists));
  PetscCheck(exists,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Restart directory %s not found.",restart_dir);

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  /* Create the output directory */
  PetscCall(PetscTestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->output_dir); }

  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-restart_step",&ctx->step,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -restart_step not found.");

  /* Load */
  //if (rank == 0) {
    PetscCall(SwarmInterpLoadFromFile(ctx, swarm_interp, restart_dir, PETSC_TRUE));
  //}
  //PetscCall(DMSwarmMigrate(swarm_interp,PETSC_TRUE));
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  /* Write */
  PetscCall(SwarmOutput(ctx,swarm_interp,1,output_fields,PETSC_FALSE));

  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  //PetscCall(Test());
  //PetscCall(Test_Parallel());
  //PetscCall(Test_Parallel_2());
  //PetscCall(Test_3());
  //PetscCall(Test_4());
  //PetscCall(TestMemoryUsage());
  //PetscCall(Test_CellInfo());
  PetscCall(Test_Load());
  PetscCall(PetscFinalize());
  
  return 0;
}