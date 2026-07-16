lagrbf_fill_swarm
=================

.. c:function:: PetscErrorCode lagrbf_fill_swarm( void *ctx, PetscBool is_data, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx )

   Fill the data or interpolation swarm with coordinates and field values. To be called from Python.

   **Collective**

   **Input Parameters**

   - ``ctx`` (``void *``) - the LagRBFCtx context cast to void*
   - ``is_data`` (`PetscBool <https://petsc.org/release/manualpages/Sys/PetscBool/>`_) - ``PETSC_TRUE`` if filling the data swarm, ``PETSC_FALSE`` if filling the interpolation swarm
   - ``npoints`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - number of points
   - ``coords`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array of coordinates of size ``npoints*dim``
   - ``values`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array of field values of size ``npoints*nfields`` (can be ``NULL`` if no field values are provided)
   - ``pidx`` (``PetscInt64 *``) - array of unique point indices of size ``npoints`` (can be ``NULL`` if not provided)
   - ``fieldname`` (``const char *``) - name of the field to fill (only used if values is not ``NULL``)

   **Output Parameters**

   - ``ierr`` (`PetscErrorCode <https://petsc.org/release/manualpages/Sys/PetscErrorCode/>`_) - 0 if successful, PETSc error code if an error occurred

   **Source**

   - ``src/interface/interface_utils.c``
