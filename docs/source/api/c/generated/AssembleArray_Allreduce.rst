AssembleArray_Allreduce
=======================

.. c:function:: PetscErrorCode AssembleArray_Allreduce( const void *local_array, PetscInt nlocal, void *global_array, PetscInt nglobal, const PetscInt64 *gidx, PetscInt nfields, PetscDataType array_type, MPI_Comm comm )

   Assemble a global array from local arrays across all ranks using MPI_Allreduce.

   **Collective**

   **Input Parameters**

   - ``local_array`` (``const void *``) - local (rank owned) array of size ``nlocal*nfields``
   - ``nlocal`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - number of points in the local array
   - ``global_array`` (``void *``) - global array of size ``nglobal*nfields`` (output), all ranks will have the assembled array
   - ``nglobal`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - total number of points in the global array
   - ``gidx`` (``const PetscInt64 *``) - global point IDs corresponding to the local points (size ``nlocal``)
   - ``nfields`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - number of fields per point
   - ``array_type`` (``PetscDataType``) - data type of the array (PETSC_REAL, PETSC_INT, or PETSC_INT64)
   - ``comm`` (``MPI_Comm``) - MPI communicator

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function assembles a global array from local arrays across all ranks using MPI_Allreduce. Each rank contributes its local array to the global array based on the provided global point IDs. The global array is summed across all ranks, and each rank will have the fully assembled global array after the operation.

   **Source**

   - ``src/interface/interface_utils.c``
