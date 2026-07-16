FillSwarm
=========

.. c:function:: PetscErrorCode FillSwarm( LagRBFCtx ctx, DM sw, PetscInt npoints, PetscReal *coords, PetscReal *values, PetscInt64 *pidx )

   Fill a DMSwarm with coordinates, field values and point indices.

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - the LagRBFCtx context
   - ``sw`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the DMSwarm to fill
   - ``npoints`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - number of points
   - ``coords`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array of coordinates of size ``npoints*dim``
   - ``values`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - array of field values of size ``npoints*nfields`` (can be ``NULL`` if no field values are provided)
   - ``pidx`` (``PetscInt64 *``) - array of unique point indices of size ``npoints`` (can be ``NULL`` if not provided and will be assigned automatically)

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function fills the DMSwarm with the provided coordinates, field values and point indices. If point indices are not provided, unique indices will be assigned automatically. The swarm is migrated across ranks to ensure that points are on the correct ranks according to the spatial decomposition.

   **Source**

   - ``src/interface/interface_utils.c``
