LagRBF_PartitionsReset
======================

.. c:function:: PetscErrorCode LagRBF_PartitionsReset(LagRBFCtx ctx)

   Reset the partitions to their initial state, clearing any computed coefficients and data point indices.

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - the LagRBF context

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function resets the partitions based on the solve method specified in the LagRBFCtx context. It clears any computed coefficients and data point indices, preparing the partitions for a new setup or solve.

   **Source**

   - ``src/partitions/partition.c``
