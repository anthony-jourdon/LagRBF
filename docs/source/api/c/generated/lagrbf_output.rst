lagrbf_output
=============

.. c:function:: PetscErrorCode lagrbf_output( void *ctx, PetscBool is_data, PetscInt npoints, PetscReal *coords, PetscReal *fields, PetscInt64 *pidx , const char *fieldname )

   Gather and assemble data from a DMSwarm across all ranks using MPI_Allreduce. To be called from Python.

   **Collective**

   **Input Parameters**

   - ``ctx`` (``void *``) - the LagRBFCtx context cast to void*
   - ``is_data`` (`PetscBool <https://petsc.org/release/manualpages/Sys/PetscBool/>`_) - ``PETSC_TRUE`` if gathering from the data swarm, ``PETSC_FALSE`` if gathering from the interpolation swarm
   - ``npoints`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - total number of points in the global array
   - ``coords`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array to store global coordinates of size ``npoints*dim`` (can be ``NULL`` if not needed)
   - ``fields`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array to store global field values of size ``npoints*nfields`` (can be ``NULL`` if not needed)
   - ``pidx`` (``PetscInt64 *``) - array to store global point indices of size ``npoints`` (can be ``NULL`` if not needed)
   - ``fieldname`` (``const char *``) - name of the field to gather (only used if fields is not ``NULL``)

   **Output Parameters**

   - ``ierr`` (`PetscErrorCode <https://petsc.org/release/manualpages/Sys/PetscErrorCode/>`_) - 0 if successful, PETSc error code if an error occurred

   **Notes**

   - This function gathers and assembles data from a DMSwarm across all ranks using MPI_Allreduce. Calls :c:func:`Interface_Output_Allreduce()`.
   - The user is responsible for allocating the output arrays (coords, fields, pidx) with the correct sizes before calling this function.

   **Source**

   - ``src/interface/interface_utils.c``
