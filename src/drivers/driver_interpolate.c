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
  file:    driver_interpolate.c

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

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "io_utils.h"
#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "swarm/swarm_output.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_load.h"
#include "readers/reader_utils.h"
#include "readers/petsc_loader/petsc_loader.h"
#include "interpolation.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "cell_info.h"
#include "time_loop.h"

static PetscErrorCode SwarmInterpSetCoords(LagRBFCtx ctx, DM sw)
{
  PetscBool  load;
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(DMGetOptionsPrefix(sw,&prefix));
  
  load = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,prefix,"-load",&load,NULL));
  if (load) {
    /* Load on swarm on rank 0 */
    PetscCall(PetscLoadToSwarmFromFiles(ctx,sw));
    /* Send to other ranks */
    PetscCall(DMSwarmAssignMigrateBasic(ctx, sw));
    /* Set back the migrate type to DMCELLNSCATTER */
    PetscCall(DMSwarmSetMigrateType(sw,DMSWARM_MIGRATE_DMCELLNSCATTER));
    /* Set cell IDs */
    PetscCall(DMSwarmPICSetCellId(sw));
  } else {
    PetscCall(DMSwarmSetPointsCoordinates_FromOptions(sw, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode driver_interpolate()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  CellInfo         *ci;
  DM               swarm_data,swarm_interp;
  PetscMPIInt      rank;
  PetscInt         step;
  PetscBool        found;
  const char       *prefix_data;
  PetscFunctionBeginUser;
  
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Create context */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  /* Create the output directory */
  PetscCall(CreateDirectory(ctx->output_dir));
  
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Create the swarm to hold data */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(ctx,&swarm_interp));

  /* Set the coordinates for the interpolation swarm */
  PetscCall(SwarmInterpSetCoords(ctx, swarm_interp));

  /* Load data */
  PetscCall(DMGetOptionsPrefix(swarm_data,&prefix_data));
  PetscCall(PetscOptionsGetInt(NULL,prefix_data,"-step",&step,&found));
  if (found) { PetscCall(LagRBFSetStep(ctx,step)); }
  PetscCall(SwarmDataLoad(ctx, swarm_data));

  /* Set weights for interpolation */
  PetscCall(SwarmInterpSetWeights(ctx,swarm_interp));
  /* Solve interpolation problem */
  PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
  /* Interpolate */
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,ctx->noutput,(const char **)(ctx->output_fields),PETSC_FALSE));

  /* Clean up */
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,NULL));
  PetscCall(driver_interpolate());
  PetscCall(PetscFinalize());
  return 0;
}