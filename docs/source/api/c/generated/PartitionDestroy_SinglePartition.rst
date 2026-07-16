PartitionDestroy_SinglePartition
================================

.. c:function:: PetscErrorCode PartitionDestroy_SinglePartition(Partition *partition, PetscErrorCode (*destroy)(Partition))

   Destroy a single partition structure.

   **Collective**

   **Input Parameters**

   - ``partition`` (:c:type:`Partition`) - pointer to the Partition structure to destroy

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function frees the memory allocated for a single partition structure.
   - It is called by :c:func:`PartitionDestroy()` for each partition.

   **See also**

   - :c:func:`PartitionDestroy`
   - :c:func:`PartitionCreate_SinglePartition`
   - :c:func:`PartitionCreate`

   **Source**

   - ``src/partitions/partition.c``
