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
  file:    swarm_load.c

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
#include "partitions/partition.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_load.h"
#include "cell_info.h"
#include "io_utils.h" 

PetscErrorCode SwarmLoadVecFromFile(DM sw, const char *field_name, const char *filename, MPI_Comm comm, PetscBool resize)
{
  Vec       vec;
  PetscInt  bs,size,npoints;
  PetscReal *pvalues,*values;
  PetscFunctionBeginUser;

  /* Load vector from file */
  PetscCall(VecCreate(comm, &vec));
  PetscCall(VecLoadFromFile(vec, filename));
  PetscCall(VecGetBlockSize(vec,&bs));
  PetscCall(VecGetSize(vec,&size));
  npoints = (PetscInt)(size / bs);
  PetscPrintf(PETSC_COMM_WORLD,"  Loaded %" PetscInt_FMT " points of block size %" PetscInt_FMT " from file: %s\n",npoints,bs,filename);
  if (resize) {
    /* Set swarm size */
    PetscCall(DMSwarmSetLocalSizes(sw,npoints,-1));
  } else {
    /* Check that the loaded vector size matches the swarm size */
    PetscInt swarm_npoints;
    PetscCall(DMSwarmGetLocalSize(sw,&swarm_npoints));
    PetscCheck(npoints == swarm_npoints, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "[[ %s ]] Loaded vector size %" PetscInt_FMT " does not match swarm size %" PetscInt_FMT "", PETSC_FUNCTION_NAME, npoints, swarm_npoints);
  }
  /* Get swarm field */
  PetscCall(DMSwarmGetField(sw, field_name, NULL, NULL, (void **)&pvalues));
  /* Copy values from vector to swarm */
  PetscCall(VecGetArray(vec,&values));
  PetscCall(PetscArraycpy(pvalues,values,size));
  PetscCall(VecRestoreArray(vec,&values));
  /* Clean up */
  PetscCall(VecDestroy(&vec));
  PetscCall(DMSwarmRestoreField(sw, field_name, NULL, NULL, (void **)&pvalues));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmLoadCoordinatesFromFile(DM sw, const char *filename, MPI_Comm comm)
{
  DMSwarmType   sw_type;
  DMSwarmCellDM celldm;
  const char    **coordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  /* Ensure the DM swarm is a PIC swarm */
  PetscCall(DMSwarmGetType(sw, &sw_type));
  PetscCheck(sw_type == DMSWARM_PIC, PETSC_COMM_SELF, PETSC_ERR_SUP, "Loading coordinates from file is only supported for DMSWARM_PIC type");

  /* Load coordinates from file */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(SwarmLoadVecFromFile(sw,coordFields[0], filename, comm, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmLoadPidxFromFile(DM sw, const char *filename, MPI_Comm comm)
{
  PetscLayout    map;
  IS             pid;
  PetscInt       np,N,npoints;
  const PetscInt *is_pidx;
  PetscInt64     *pidx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Load the point indices stored as Index Set (IS) */
  PetscCall(ISCreate(comm, &pid));
  PetscCall(ISSetType(pid,ISGENERAL));
  PetscCall(ISLoadFromFile(pid, filename));
  /* Check that IS is the same size than the swarm */
  PetscCall(ISGetLayout(pid, &map));
  PetscCall(PetscLayoutGetSize(map, &N));
  PetscCall(DMSwarmGetLocalSize(sw, &npoints));
  PetscCheck(N == npoints, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "[[ %s ]] Loaded point index size %" PetscInt_FMT " does not match swarm size %" PetscInt_FMT "", PETSC_FUNCTION_NAME, N, npoints);
  /* Copy point indices into the swarm */
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  PetscCall(ISGetIndices(pid, &is_pidx));
  for (np=0; np<npoints; np++) { pidx[np] = (PetscInt64)is_pidx[np]; }
  PetscCall(ISRestoreIndices(pid, &is_pidx));
  /* Clean up */
  PetscCall(ISDestroy(&pid));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscErrorCode SwarmLoadFromFiles(LagRBFCtx ctx, DM sw, const char *dirname, const char **fields_name, PetscInt nfields)
  
  Load the swarm coordinates and point indices from files located in 'dirname'.
  The files are expected to be named according to the convention:
    <stepprefix>.<dm_name>_<coord_field>.pbvec  for coordinates
    <stepprefix>.<dm_name>_pid.pbis            for point indices
  where <stepprefix> is derived from the current step in the LagRBFCtx context and potentially appended with "_forward" or "_backward" if advection is active.
  The point loading is only performed on rank 0, followed by a broadcast to other ranks using a basic migration strategy.
  The switch to a basic migration strategy is necessary because DMSWARM_MIGRATE_DMCELLNSCATTER only uses neighbor ranks for point migration,
  which can result in incomplete data distribution.
  After broadcasting, the migrate type is reset to DMSWARM_MIGRATE_DMCELLNSCATTER and cell IDs are assigned using the DMSwarmCellDM.
*/
PetscErrorCode SwarmLoadFromFiles(LagRBFCtx ctx, DM sw, const char *dirname, const char **fields_name, PetscInt nfields)
{
  DMSwarmType   sw_type;
  DMSwarmCellDM celldm;
  PetscInt      step,f;
  MPI_Comm      comm;
  PetscMPIInt   rank;
  char          filename[PETSC_MAX_PATH_LEN],stepprefix[PETSC_MAX_PATH_LEN];
  const char    *dm_name,**coordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmGetType(sw,&sw_type));
  PetscCheck(sw_type == DMSWARM_PIC, PETSC_COMM_SELF, PETSC_ERR_SUP, "Loading from file is only supported for DMSWARM_PIC type");
  
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscObjectGetName((PetscObject)sw,&dm_name));
  PetscCall(LagRBFGetStep(ctx,&step));
  PetscCall(PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%06" PetscInt_FMT "",step));
  if (ctx->forward)  { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_forward", step)); }
  if (ctx->backward) { PetscCall(PetscSNPrintf(stepprefix, PETSC_MAX_PATH_LEN-1, "step%06" PetscInt_FMT "_backward", step)); }
  
  /* Load coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(PetscSNPrintf(filename, PETSC_MAX_PATH_LEN-1, "%s/%s.%s_%s.pbvec", dirname, stepprefix, dm_name, coordFields[0]));
  if (rank == 0) {
    /* First we load the coordinates and set the number of points */
    PetscCall(SwarmLoadCoordinatesFromFile(sw, filename, PETSC_COMM_SELF));
    /* Then we load any field required by the user */
    for (f=0; f<nfields; f++) {
      PetscDataType type;
      void          *data;
      char          field_filename[PETSC_MAX_PATH_LEN];

      /* Get the data type */
      PetscCall(DMSwarmGetField(sw,fields_name[f],NULL,&type,&data));
      PetscCall(DMSwarmRestoreField(sw,fields_name[f],NULL,&type,&data));

      switch (type) {
        case PETSC_REAL:
          PetscCall(PetscSNPrintf(field_filename, PETSC_MAX_PATH_LEN-1, "%s/%s.%s_%s.pbvec", dirname, stepprefix, dm_name, fields_name[f]));
          PetscCall(SwarmLoadVecFromFile(sw, field_filename, filename, PETSC_COMM_SELF, PETSC_FALSE));
          break;
        
        //case PETSC_INT: // For now we don't deal with other integer types than int64
        case PETSC_INT64:
          PetscCall(PetscSNPrintf(field_filename, PETSC_MAX_PATH_LEN-1, "%s/%s.%s_%s.pbis", dirname, stepprefix, dm_name, fields_name[f]));
          PetscCall(SwarmLoadPidxFromFile(sw, field_filename, PETSC_COMM_SELF));
          break;

        default:
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_UNKNOWN_TYPE,"Data type must be PETSC_REAL or PETSC_INT");
          break;
      }
    }
  }
  /* Ensure rank 0 finished loading before proceeding */
  PetscCall(PetscBarrier((PetscObject)sw));
  /* Send to other ranks */
  PetscCall(DMSwarmAssignMigrateBasic(ctx, sw));
  /* Set back the migrate type to DMCELLNSCATTER */
  PetscCall(DMSwarmSetMigrateType(sw,DMSWARM_MIGRATE_DMCELLNSCATTER));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}
