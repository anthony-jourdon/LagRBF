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
  file:    petsc_loader.c

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
#include "io_utils.h" 
#include "cell_info.h"
#include "partition_of_unity.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_load.h"
#include "swarm/swarm_data.h"
#include "readers/petsc_loader/petsc_loader.h"

PetscErrorCode SwarmSetFieldFromPetscFile(DM sw, const char *fname, PetscInt *block, MPI_Comm comm)
{
  Vec       vec;
  PetscInt  npoints,np,bs,size,np_vec,pos,nfields;
  PetscReal *pvalues,*values;
  PetscFunctionBeginUser;

  /* Index of the field to set */
  pos = *block;
  /* Get swarm size */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  /* Load vector from file */
  PetscCall(VecCreate(comm, &vec));
  PetscCall(VecLoadFromFile(vec, fname));
  PetscCall(VecGetBlockSize(vec,&bs));
  PetscCall(VecGetSize(vec,&size));
  np_vec = (PetscInt)(size / bs);
  if (npoints <= 0) {
    /* Set swarm size based on the first loaded vector */
    PetscCall(DMSwarmSetLocalSizes(sw,np_vec,-1));
    npoints = np_vec;
  }
  /* Make sure that the loaded vector size matches the swarm size */
  PetscCheck(np_vec == npoints, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "[[ %s ]] Loaded vector size %" PetscInt_FMT " does not match swarm size %" PetscInt_FMT " for field from file %s", PETSC_FUNCTION_NAME, np_vec, npoints, fname);
  PetscPrintf(PETSC_COMM_WORLD,"  Loaded %" PetscInt_FMT " points of block size %" PetscInt_FMT " from file: %s\n", np_vec, bs, fname);

  /* Fill swarm */
  PetscCall(DMSwarmGetField(sw, "fields", &nfields, NULL, (void **)&pvalues));
  /* Make sure that the field position is valid */
  PetscCheck(pos + bs <= nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "[[ %s ]] Too many fields to load. Loaded fields require block size %" PetscInt_FMT " but only %" PetscInt_FMT " fields are available in the swarm.", PETSC_FUNCTION_NAME, pos + bs, nfields);
  PetscCall(VecGetArray(vec,&values));
  for (np=0; np<npoints; np++) {
    PetscInt b;
    for (b=0; b<bs; b++) {
      pvalues[np*nfields + pos + b] = values[np*bs + b]; 
    }
  }
  *block += bs;
  PetscCall(DMSwarmRestoreField(sw, "fields", &nfields, NULL, (void **)&pvalues));
  PetscCall(VecRestoreArray(vec,&values));
  PetscCall(VecDestroy(&vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmSetFromPetscFiles(DM sw, const char **filenames, PetscInt nfiles, const char *coor_fname)
{
  PetscInt f,block;
  char     fname[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  /* First we load the coordinates and set the number of points */
  PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s.pbvec", coor_fname));
  PetscCall(SwarmLoadCoordinatesFromFile(sw, fname, PETSC_COMM_SELF));

  /* Then we load any field required by the user */
  block = 0;
  for (f=0; f<nfiles; f++) {
    PetscCall(PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s.pbvec", filenames[f]));
    PetscCall(SwarmSetFieldFromPetscFile(sw,fname,&block,PETSC_COMM_SELF));
    
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLoadToSwarmFromFiles(LagRBFCtx ctx, DM sw)
{
  PetscMPIInt      rank;
  PetscInt         nfiles,step;
  char             **filenames;
  char             *coor_fname;
  const char       *prefix;
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetStep(ctx,&step));

  PetscCall(DMGetOptionsPrefix(sw, &prefix));
  PetscCall(GetFilelistFromOptions(prefix,step,&nfiles,&filenames,&coor_fname));
  /* Read data only on rank 0 */
  if (rank == 0) {
    PetscCall(SwarmSetFromPetscFiles(sw, (const char **)filenames, nfiles, (const char *)coor_fname));
  }
  /* Ensure that rank 0 has finished reading data */
  PetscCall(PetscBarrier((PetscObject)NULL));
  /* Free filenames */
  PetscCall(RestoreFilelist(nfiles,&filenames,&coor_fname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLoadDataToSwarmFromFiles(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&ci));

  /* Load on swarm */
  PetscCall(PetscLoadToSwarmFromFiles(ctx,sw));

  /* Send data to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx,sw));
  /* Duplicate ghost points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,sw));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  /* Set up CellInfo data structure */
  PetscCall(CellInfoSetUp(pu,ci,sw));

  PetscFunctionReturn(PETSC_SUCCESS);
}