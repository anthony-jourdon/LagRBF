lagrbf_reset_coefficients
=========================

.. c:function:: PetscErrorCode lagrbf_reset_coefficients(void *ctx)

   Reset the coefficients of the RBF interpolation to zero. To be called from Python.

   **Collective**

   **Input Parameters**

   - ``ctx`` (``void *``) - the LagRBFCtx context cast to void*

   **Output Parameters**

   - ``ierr`` (`PetscErrorCode <https://petsc.org/release/manualpages/Sys/PetscErrorCode/>`_) - 0 if successful, PETSc error code if an error occurred

   **Notes**

   - This function resets the coefficients of the RBF interpolation to zero. User must call :c:func:`lagrbf_reset_coefficients` when they want to reset the coefficients.
   - This function is a wrapper for :c:func:`LagRBF_PartitionsReset` to be called from Python.

   **Source**

   - ``src/interface/interface_utils.c``
