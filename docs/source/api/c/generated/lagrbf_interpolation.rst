lagrbf_interpolation
====================

.. c:function:: PetscErrorCode lagrbf_interpolation(void *ctx)

   Interpolate values of one field from data swarm points to interpolation swarm points. To be called from Python.

   **Collective**

   **Input Parameters**

   - ``ctx`` (``void *``) - the LagRBFCtx context cast to void*

   **Output Parameters**

   - ``ierr`` (`PetscErrorCode <https://petsc.org/release/manualpages/Sys/PetscErrorCode/>`_) - 0 if successful, PETSc error code if an error occurred

   **Notes**

   - Wrapper for :c:func:`Interface_Interpolation` to be called from Python.

   **Source**

   - ``src/interface/interface_interp.c``
