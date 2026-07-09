#include "lagrangian_rbf.h"
#include "lagrangian_rbf_init.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "cell_info.h"
#include "partitions/partition.h"
#include "interpolation.h"
#include "interface/interface_utils.h"

PetscErrorCode InterfacePetscErrorPrinter(PetscErrorCode ierr, const char *function_name)
{
  const char *err_str;
  char       *specific;
  PetscFunctionBeginUser;
  if (!ierr) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(PetscErrorMessage(ierr, &err_str, &specific));
  PetscPrintf(PETSC_COMM_WORLD, "ERROR: %s() failed with error code %d: %s\nERROR: %s\n", function_name, ierr, err_str, specific ? specific : "");
  PetscFunctionReturn(ierr);
}

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

PetscErrorCode FillInterpSwarm(
  LagRBFCtx   ctx, 
  PetscInt    npoints,
  PetscReal  *coords, 
  PetscReal  *values,
  PetscInt64 *pidx
)
{
  DM sw;
  PetscFunctionBeginUser;
  sw = ctx->swi;
  /* Fill the swarm with input arrays */
  PetscCall(FillSwarm(ctx, sw, npoints, coords, values, pidx));
  /* Set back the migrate type to DMCELLNSCATTER */
  PetscCall(DMSwarmSetMigrateType(sw, DMSWARM_MIGRATE_DMCELLNSCATTER));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  /* Compute weights for partition of unity method */
  PetscCall(SwarmInterpSetWeights(ctx, sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FillDataSwarm(
  LagRBFCtx   ctx, 
  PetscInt    npoints,
  PetscReal  *coords, 
  PetscReal  *values,
  PetscInt64 *pidx
)
{
  PartitionOfUnity pu;
  CellInfo         *ci;
  DM               sw;
  PetscFunctionBeginUser;
  sw = ctx->swd;
  /* Fill the swarm with input arrays */
  PetscCall(FillSwarm(ctx, sw, npoints, coords, values, pidx));
  /* Duplicate ghost data points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx, sw));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(sw));
  /* Set up CellInfo data structure */
  PetscCall(LagRBFGetPartitionOfUnity(ctx, &pu));
  PetscCall(LagRBFGetCellInfo(ctx, &ci));
  PetscCall(CellInfoSetUp(pu, ci, sw));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleArray_Allreduce(
  const void       *local_array,  // local (rank owned) array of size nlocal*nfields
  PetscInt         nlocal,        // number of points in the local array 
  void             *global_array, // global array of size nglobal*nfields (output), all ranks have it
  PetscInt         nglobal,       // total number of points in the global array
  const PetscInt64 *gidx,         // global point IDs corresponding to the local points (size nlocal)
  PetscInt         nfields,       // number of fields per point
  PetscDataType    array_type,    // PETSC_REAL, PETSC_INT or PETSC_INT64
  MPI_Comm         comm           // communicator
)
{
  PetscInt     np,f,nentries;
  PetscMPIInt  nentries_mpi;
  size_t       elem_size;
  MPI_Datatype mpi_type;
  PetscFunctionBeginUser;

  PetscCheck(local_array,  PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "[[ %s ]] local_array is NULL",  PETSC_FUNCTION_NAME);
  PetscCheck(global_array, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "[[ %s ]] global_array is NULL", PETSC_FUNCTION_NAME);
  PetscCheck(gidx,         PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "[[ %s ]] gidx is NULL",         PETSC_FUNCTION_NAME);

  switch (array_type) {
    case PETSC_REAL:
      elem_size = sizeof(PetscReal);
      mpi_type  = MPIU_REAL;
      break;
    case PETSC_INT:
      elem_size = sizeof(PetscInt);
      mpi_type  = MPIU_INT;
      break;
    case PETSC_INT64:
      elem_size = sizeof(PetscInt64);
      mpi_type  = MPIU_INT64;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported array_type %d", (int)array_type);
  }

  nentries = nglobal * nfields;
  /* Zero the global array because we will be summing into it */
  PetscCall(PetscMemzero(global_array, (size_t)nentries * elem_size));

  for (np=0; np<nlocal; np++) {
    PetscInt64 pid = gidx[np]-1; // convert to 0-based index
    PetscInt   pid32;
    PetscCheck(
      pid >= 0 && pid < nglobal, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "pid %" PetscInt64_FMT " out of range [0, %" PetscInt_FMT ")", pid, nglobal
    );
    PetscCall(PetscIntCast(pid, &pid32));
    /* Each rank fills its portion of the global array */
    for (f=0; f<nfields; f++) {
      PetscInt idx_global,idx_local;
      idx_global = pid * nfields + f;
      idx_local  = np  * nfields + f;
      switch (array_type) {
        case PETSC_REAL:
          ((PetscReal *)global_array)[idx_global] = ((const PetscReal *)local_array)[idx_local];
          break;
        case PETSC_INT:
          ((PetscInt *)global_array)[idx_global] = ((const PetscInt *)local_array)[idx_local];
          break;
        case PETSC_INT64:
          ((PetscInt64 *)global_array)[idx_global] = ((const PetscInt64 *)local_array)[idx_local];
          break;
        default:
          break;
      }
    }
  }
  PetscCall(PetscMPIIntCast(nentries, &nentries_mpi));
  /* Perform reduction to assemble the global array */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, global_array, nentries_mpi, mpi_type, MPI_SUM, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Interface_Output_Allreduce(
  DM          sw,
  PetscInt    np_global,
  PetscReal  *global_coor,
  PetscReal  *global_fields,
  PetscInt64 *global_pidx,
  const char *fieldname
)
{
  PetscInt   np_local;
  PetscInt64 *local_pidx;
  PetscFunctionBeginUser;
  
  PetscCall(DMSwarmGetLocalSize(sw,&np_local));

  /* Get the unique point indices */
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&local_pidx));

  /* coordinates */
  if (global_coor != NULL) {
    DMSwarmCellDM celldm;
    PetscInt      dim;
    PetscReal     *local_coor;
    const char    **coordFieldName;

    PetscCall(DMGetDimension(sw,&dim));
    PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
    PetscCall(DMSwarmCellDMGetCoordinateFields(celldm,NULL,&coordFieldName));
    PetscCall(DMSwarmGetField(sw, coordFieldName[0], NULL, NULL, (void **)&local_coor));
    /* Assemble */
    PetscCall(AssembleArray_Allreduce(local_coor, np_local, global_coor, np_global, local_pidx, dim, PETSC_REAL, PETSC_COMM_WORLD));
    PetscCall(DMSwarmRestoreField(sw, coordFieldName[0], NULL, NULL, (void **)&local_coor));
  }

  /* fields */
  if (global_fields != NULL) {
    PetscInt nfields;
    PetscReal *local_fields;

    PetscCall(DMSwarmGetField(sw, fieldname, &nfields, NULL, (void **)&local_fields));
    /* Assemble */
    PetscCall(AssembleArray_Allreduce(local_fields, np_local, global_fields, np_global, local_pidx, nfields, PETSC_REAL, PETSC_COMM_WORLD));
    PetscCall(DMSwarmRestoreField(sw, fieldname, &nfields, NULL, (void **)&local_fields));
  }

  /* pidx */
  if (global_pidx != NULL) {
    /* Assemble */
    PetscCall(AssembleArray_Allreduce(local_pidx, np_local, global_pidx, np_global, local_pidx, 1, PETSC_INT64, PETSC_COMM_WORLD));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&local_pidx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_output(
  void       *ctx,
  PetscBool   is_data,
  PetscInt    npoints,
  PetscReal  *coords,
  PetscReal  *fields,
  PetscInt64 *pidx  ,
  const char *fieldname
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  DM             sw;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if (is_data) { sw = user->swd; }
  else         { sw = user->swi; }
  ierr = Interface_Output_Allreduce(sw,npoints,coords,fields,pidx,fieldname);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "Interface_Output_Allreduce");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_fill_swarm(
  void       *ctx,
  PetscBool   is_data,
  PetscInt    npoints,
  PetscReal  *coords,
  PetscReal  *values,
  PetscInt64 *pidx
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  /* Initialize to empty character */
  fname[0] = '\0';
  if (is_data) {
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"FillDataSwarm");
    ierr = FillDataSwarm(user, npoints, coords, values, pidx);
  } else {
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"FillInterpSwarm");
    ierr = FillInterpSwarm(user, npoints, coords, values, pidx);
  }
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, fname);
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}