PartitionCreate_SinglePartition
===============================

.. c:function:: PetscErrorCode PartitionCreate_SinglePartition(Partition *partition)

   Create a single partition structure.

   **Collective**

   **Input Parameters**

   - ``partition`` (:c:type:`Partition`) - pointer to the Partition structure to create

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function allocates memory for a single partition structure and initializes its members to default values.
   - Memory is freed in :c:func:`PartitionDestroy_SinglePartition()` for a given partition.

   **See also**

   - :c:func:`PartitionCreate`
   - :c:func:`PartitionDestroy_SinglePartition`
   - :c:func:`PartitionDestroy`

   **Source**

   - ``src/partitions/partition.c``
