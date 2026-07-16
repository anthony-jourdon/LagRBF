PartitionCreate
===============

.. c:function:: PetscErrorCode PartitionCreate(PartitionOfUnity pu, Partition **partition)

   Create and initialize all partitions.

   **Collective**

   **Input Parameters**

   - ``pu`` (:c:type:`PartitionOfUnity`) - the PartitionOfUnity context
   - ``partition`` (:c:type:`Partition`) - array of Partition structures

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function creates and initializes all partitions of the domain.
   - It allocates memory for each partition and sets up the inverse mapping from partitions to cells.
   - Memory is freed in :c:func:`PartitionDestroy()` for all partitions.

   **See also**

   - :c:func:`PartitionDestroy`
   - :c:func:`PartitionCreate_SinglePartition`
   - :c:func:`PartitionDestroy_SinglePartition`
   - :c:func:`PartitionCreate_PartitionToCellInverseMapping`

   **Source**

   - ``src/partitions/partition.c``
