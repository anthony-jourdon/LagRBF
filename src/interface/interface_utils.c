#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "cell_info.h"
#include "partitions/partition.h"
#include "interpolation.h"

PetscErrorCode FillSwarm(
  LagRBFCtx  ctx, 
  DM         sw, 
  PetscInt   npoints, 
  PetscReal  *coords, 
  PetscReal  *values,
  PetscInt64 *pidx
)
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
  /* If point indices are provided use them, otherwise assign unique indices */
  if (pidx != NULL) {
    PetscInt64 *swarm_pidx;
    PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&swarm_pidx));
    PetscCall(PetscArraycpy(swarm_pidx, pidx, npoints));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&swarm_pidx));
  } else {
    PetscCall(SwarmFieldAssignUniqueIndex(sw,DMSwarmField_pid, 0, npoints));
  }
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