LagRBF_PartitionsSetUp
======================

.. c:function:: PetscErrorCode LagRBF_PartitionsSetUp(LagRBFCtx ctx, DM swd, DM swi)

   Set up all partitions and solve the coefficients for interpolation.

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - the LagRBF context
   - ``swd`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the data swarm DM
   - ``swi`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the interpolation swarm DM

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function sets up all partitions and solves for the coefficients needed for interpolation.
   - It retrieves the necessary information from the PartitionOfUnity context and the swarms, and iterates over each point in the interpolation swarm to set up and solve the corresponding partitions.
   - Partitions are set up and solved only once, even if multiple points are connected to the same partition. Thus the necessity to reset them with :c:func:`PartitionReset()` between calls to this function if the data swarm changes.

   **See also**

   - :c:func:`PartitionReset`

   **Source**

   - ``src/partitions/partition.c``
