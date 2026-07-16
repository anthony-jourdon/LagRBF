PartitionCreate_PartitionToCellInverseMapping
=============================================

.. c:function:: PetscErrorCode PartitionCreate_PartitionToCellInverseMapping(PartitionOfUnity pu, Partition *partition)

   Create the inverse mapping from partition to cells.

   **Collective**

   **Input Parameters**

   - ``pu`` (:c:type:`PartitionOfUnity`) - the PartitionOfUnity context
   - ``partition`` (:c:type:`Partition`) - array of Partition structures

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - Create the inverse map from partition to cells.
   - The partition is a node of the DMDA, and the cells are the elements of the DMDA.
   - The map is an array of indices, where each index corresponds to a cell that is connected to the partition.
   - The number of cells connected to each partition is stored in ``ncells``.
   - A map array is allocated for each partition and filled with the cell indices.
   - Memory is freed in :c:func:`PartitionDestroy_SinglePartition()` for a given partition or in :c:func:`PartitionDestroy()` for all partitions.

   **See also**

   - :c:func:`PartitionCreate`

   **Source**

   - ``src/partitions/partition.c``
