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
  PetscMPIInt      rank;
  const char       **coordFields;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    PetscCall(DMGetDimension(sw,&dim));
    /* Set swarm size */
    PetscCall(DMSwarmSetLocalSizes(sw,npoints,-1));
    /* Fill coordinates */
    /* Get swarm coordinates field */
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
  /* Ensure that rank 0 has finished writing data */
  PetscCall(PetscBarrier((PetscObject)NULL));
  /* Send data to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx, sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LagRBF_Interpolate_Interface(
  PetscInt  ndata,
  PetscReal *data_coords,
  PetscReal *data_values,
  PetscInt  npoints,
  PetscReal *points_coords,
  PetscReal *points_values,
  char      *options_file
)
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscInt         nfields;
  PetscReal        *fvals;
  PetscFunctionBeginUser;
  /* Insert options from file */
  PetscCall(LagRBFInitialize(0,NULL,options_file,NULL));

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
  PetscCall(FillSwarm(ctx, swarm_data,   ndata,   data_coords,   data_values));
  PetscCall(FillSwarm(ctx, swarm_interp, npoints, points_coords, NULL));
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

  /* Copy back interpolated values */
  PetscCall(DMSwarmGetField(swarm_interp,"fields",&nfields,NULL,(void **)&fvals));
  PetscCall(PetscArraycpy(points_values,fvals,npoints*nfields));
  PetscCall(DMSwarmRestoreField(swarm_interp,"fields",&nfields,NULL,(void **)&fvals));
  
  /* Clean up */
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));

  //PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void lagrbf_interpolate(
  int    ndata,
  double *data_coords,
  double *data_values,
  int    npoints,
  double *points_coords,
  double *points_values,
  char   *options_file
)
{
  
  LagRBF_Interpolate_Interface(
    (PetscInt)ndata,
    (PetscReal *)data_coords,
    (PetscReal *)data_values,
    (PetscInt)npoints,
    (PetscReal *)points_coords,
    (PetscReal *)points_values,
    options_file
  );
  
}