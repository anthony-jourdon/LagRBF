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
  PetscCall(SwarmOutputCoordsPetscVec(swarm_data,output_dir));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_data,output_dir));
  

  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}