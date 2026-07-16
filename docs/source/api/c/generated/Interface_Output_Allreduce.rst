Interface_Output_Allreduce
==========================

.. c:function:: PetscErrorCode Interface_Output_Allreduce( DM sw, PetscInt np_global, PetscReal *global_coor, PetscReal *global_fields, PetscInt64 *global_pidx, const char *fieldname )

   Gather and assemble data from a DMSwarm across all ranks using MPI_Allreduce.

   **Collective**

   **Input Parameters**

   - ``sw`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the DMSwarm to gather data from
   - ``np_global`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - total number of points in the global array
   - ``global_coor`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array to store global coordinates of size ``np_global*dim`` (can be ``NULL`` if not needed)
   - ``global_fields`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array to store global field values of size ``np_global*nfields`` (can be ``NULL`` if not needed)
   - ``global_pidx`` (``PetscInt64 *``) - array to store global point indices of size ``np_global`` (can be ``NULL`` if not needed)
   - ``fieldname`` (``const char *``) - name of the field to gather (only used if global_fields is not ``NULL``)

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function gathers and assembles data from a DMSwarm across all ranks using MPI_Allreduce. Calls :c:func:`AssembleArray_Allreduce()`.

   **See also**

   - :c:func:`AssembleArray_Allreduce`

   **Source**

   - ``src/interface/interface_utils.c``
