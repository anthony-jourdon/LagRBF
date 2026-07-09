#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "cell_info.h"
#include "partitions/partition.h"
#include "interpolation.h"
#include "interface/interface_interp.h"
#include "interface/interface_utils.h"

/*
  Build a full points_values array on every rank.
  Each rank writes values for the pids it owns locally, then an MPI_Allreduce
  (sum) assembles a complete global array on all ranks.
*/
static PetscErrorCode FieldsAssembleArray_Allreduce(DM swarm_interp, PetscReal *points_values, PetscInt npoints_global)
{
  MPI_Comm   comm;
  PetscInt   nlocal = 0, nfields = 0;
  PetscInt64 *pids = NULL;
  PetscReal  *fvals = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)swarm_interp, &comm));

  PetscCall(DMSwarmGetLocalSize(swarm_interp, &nlocal));
  PetscCall(DMSwarmGetField(swarm_interp, "fields", &nfields, NULL, (void **)&fvals));
  PetscCall(DMSwarmGetField(swarm_interp, DMSwarmField_pid, NULL, NULL, (void **)&pids));

  PetscCall(AssembleArray_Allreduce(fvals,nlocal,points_values,npoints_global,pids,nfields,PETSC_REAL,comm));

  PetscCall(DMSwarmRestoreField(swarm_interp, DMSwarmField_pid, NULL, NULL, (void **)&pids));
  PetscCall(DMSwarmRestoreField(swarm_interp, "fields", &nfields, NULL, (void **)&fvals));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Interface_Interpolation(LagRBFCtx ctx)
{
  DM swd = ctx->swd;
  DM swi = ctx->swi;
  PetscFunctionBeginUser;
  
  PetscCheck(swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Interpolation swarm (ctx->swi) is NULL");
  PetscCheck(swd, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Data swarm (ctx->swd) is NULL");
  /* Interpolation */
  PetscCall(LagRBF_PartitionsSetUp(ctx,swd,swi));
  PetscCall(RBFInterpolation(ctx,swi,swd,"fields"));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Interface_InterpolateDriver(
  PetscInt  ndata_local,
  PetscReal *data_coords,
  PetscReal *data_values,
  PetscInt  npoints_local,
  PetscReal *points_coords,
  PetscReal *points_values,
  PetscInt  npoints_global,
  char      *options_file
)
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscMPIInt      rank;
  PetscFunctionBeginUser;

  /* Insert options from file */
  PetscCall(LagRBFInitialize(0,NULL,options_file,NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscPrintf(PETSC_COMM_SELF, "[Rank %d]: npoints_local = %" PetscInt_FMT ", npoints_global = %" PetscInt_FMT "\n", rank, npoints_local, npoints_global);

  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Create the swarm to hold data */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));

  /* Fill swarms */
  PetscCall(FillSwarm(ctx, swarm_data,   ndata_local,   data_coords,   data_values, NULL));
  PetscCall(FillSwarm(ctx, swarm_interp, npoints_local, points_coords, NULL, NULL));
  PetscCall(DMSwarmSetMigrateType(swarm_interp,DMSWARM_MIGRATE_DMCELLNSCATTER));

  /* Duplicate ghost data points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,swarm_data));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(swarm_data));
  PetscCall(DMSwarmPICSetCellId(swarm_interp));

  /* Set up CellInfo data structure */
  PetscCall(CellInfoSetUp(pu,ci,swarm_data));
  /* Compute weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  /* Interpolation */
  PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Assemble interpolation results on all ranks (all ranks have the fully assembled array) */
  PetscCall(FieldsAssembleArray_Allreduce(swarm_interp, points_values, npoints_global));

  /* Clean up */
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

void lagrbf_interpolate_driver(
  PetscInt   ndata_local,
  PetscReal *data_coords,
  PetscReal *data_values,
  PetscInt   npoints_local,
  PetscReal *points_coords,
  PetscReal *points_values,
  PetscInt   npoints_global,
  char      *options_file
)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = Interface_InterpolateDriver(
    ndata_local,
    data_coords,
    data_values,
    npoints_local,
    points_coords,
    points_values,
    npoints_global,
    options_file
  );
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "lagrbf_interpolate");
  }
  PetscFunctionReturnVoid();
}

PetscErrorCode lagrbf_interpolation(void *ctx)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = Interface_Interpolation(user);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "Interface_Interpolation");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}