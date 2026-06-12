#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "cell_info.h"
#include "partitions/partition.h"
#include "interpolation.h"

static PetscErrorCode FillSwarm(LagRBFCtx ctx, DM sw, PetscInt npoints, PetscReal *coords, PetscReal *values)
{
  DMSwarmCellDM    celldm;
  PetscInt         dim;
  PetscReal        *pcoords;
  const char       **coordFields;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(sw,&dim));
  /* Set swarm size */
  PetscCall(DMSwarmSetLocalSizes(sw,npoints,-1));
  /* Fill swarm */
  if (npoints > 0) {
    PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
    PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
    PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
    /* Copy coordinates */
    PetscCall(PetscArraycpy(pcoords,coords,npoints*dim));
    PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
    /* Fill values */
    if (values != NULL) {
      PetscInt  nfields;
      PetscReal *fval;
      PetscCall(DMSwarmGetField(sw,"fields",&nfields,NULL,(void **)&fval));
      PetscCall(PetscArraycpy(fval,values,npoints*nfields));
      PetscCall(DMSwarmRestoreField(sw,"fields",&nfields,NULL,(void **)&fval));
    }
  }
  PetscCall(SwarmFieldAssignUniqueIndex(sw,DMSwarmField_pid, 0, npoints));
  /* Migrate: send data to their correct ranks according to spatial decomposition */
  PetscCall(DMSwarmAssignMigrateBasic(ctx, sw));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleArray_Allreduce(
  PetscReal  *local_array,  // local (rank owned) array of size nlocal*nfields
  PetscInt   nlocal,        // number of points in the local array 
  PetscReal  *global_array, // global array of size nglobal*nfields (output), all ranks have it
  PetscInt   nglobal,       // total number of points in the global array
  PetscInt64 *gidx,         // global point IDs corresponding to the local points (size nlocal)
  PetscInt   nfields,       // number of fields per point
  MPI_Comm   comm           // communicator
)
{
  PetscInt    np,f,nentries;
  PetscMPIInt nentries_mpi;
  PetscFunctionBeginUser;

  nentries = nglobal * nfields;
  /* Zero the global array because we will be summing into it */
  PetscCall(PetscArrayzero(global_array, nentries));
  for (np=0; np<nlocal; np++) {
    PetscInt64 pid = gidx[np]-1; // convert to 0-based index
    PetscCheck(
      pid >= 0 && pid < nglobal, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "pid %" PetscInt64_FMT " out of range [0, %" PetscInt_FMT ")", pid, nglobal
    );
    /* Each rank fills its portion of the global array */
    for (f=0; f<nfields; f++) {
      global_array[pid * nfields + f] = local_array[np * nfields + f];
    }
  }
  PetscCall(PetscMPIIntCast(nentries, &nentries_mpi));
  /* Perform reduction to assemble the global array */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, global_array, nentries_mpi, MPIU_REAL, MPI_SUM, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Build a full points_values array on every rank.
  Each rank writes values for the pids it owns locally, then an MPI_Allreduce
  (sum) assembles a complete global array on all ranks.
*/
static PetscErrorCode FieldsAssembleArray_Allreduce(DM swarm_interp, PetscReal *points_values, PetscInt npoints_global)
{
  MPI_Comm      comm;
  PetscInt      nlocal = 0, nfields = 0;
  PetscInt      i, f;
  PetscInt      nentries;
  PetscMPIInt   nentries_mpi;
  PetscInt64    *pids = NULL;
  PetscReal     *fvals = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)swarm_interp, &comm));

  PetscCall(DMSwarmGetLocalSize(swarm_interp, &nlocal));
  PetscCall(DMSwarmGetField(swarm_interp, "fields", &nfields, NULL, (void **)&fvals));
  PetscCall(DMSwarmGetField(swarm_interp, DMSwarmField_pid, NULL, NULL, (void **)&pids));

  PetscCall(AssembleArray_Allreduce(fvals,nlocal,points_values,npoints_global,pids,nfields,comm));

  PetscCall(DMSwarmRestoreField(swarm_interp, DMSwarmField_pid, NULL, NULL, (void **)&pids));
  PetscCall(DMSwarmRestoreField(swarm_interp, "fields", &nfields, NULL, (void **)&fvals));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LagRBF_Interpolate_Interface(
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
  PetscInt         npoints_i;
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
  PetscCall(FillSwarm(ctx, swarm_data,   ndata_local,   data_coords,   data_values));
  PetscCall(FillSwarm(ctx, swarm_interp, npoints_local, points_coords, NULL));
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

void lagrbf_interpolate(
  PetscInt    ndata_local,
  PetscReal *data_coords,
  PetscReal *data_values,
  PetscInt    npoints_local,
  PetscReal *points_coords,
  PetscReal *points_values,
  PetscInt    npoints_global,
  char   *options_file
)
{
  
  LagRBF_Interpolate_Interface(
    ndata_local,
    data_coords,
    data_values,
    npoints_local,
    points_coords,
    points_values,
    npoints_global,
    options_file
  );
  
}