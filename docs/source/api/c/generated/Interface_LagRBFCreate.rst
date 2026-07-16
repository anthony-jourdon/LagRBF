Interface_LagRBFCreate
======================

.. c:function:: PetscErrorCode Interface_LagRBFCreate(LagRBFCtx *ctx)

   Creates the :c:type:`LagRBFCtx` context and initializes the data structures from options.

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - pointer to the LagRBFCtx context to be created

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function initializes the data structures in the memory. User must call :c:func:`LagRBFCtxDestroy()` to free the memory.

   **See also**

   - :c:func:`LagRBFCtxDestroy`
   - :c:func:`lagrbf_create`
   - :c:func:`lagrbf_destroy`
   - :c:func:`LagRBFCtxCreate`
   - :c:func:`LagRBFSetFromOptions`
   - :c:func:`LagRBF_PartitionOfUnityCreate`
   - :c:func:`LagRBF_PartitionCreate`
   - :c:func:`LagRBF_CellInfoCreate`
   - :c:func:`SwarmDataCreate`
   - :c:func:`SwarmInterpCreate`

   **Source**

   - ``src/interface/interface_ctx.c``
