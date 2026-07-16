PartitionFillDataPointsIdx
==========================

.. c:function:: PetscErrorCode PartitionFillDataPointsIdx(Partition p, DM swarm)

   Fill the data points indices for a partition.

   **Collective**

   **Input Parameters**

   - ``p`` (:c:type:`Partition`) - the Partition data structure
   - ``swarm`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the swarm data structure

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function fills the data points indices for a partition by retrieving the points associated with each cell connected to the partition.

   **Source**

   - ``src/partitions/partition.c``
