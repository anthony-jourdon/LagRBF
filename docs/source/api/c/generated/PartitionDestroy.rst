PartitionDestroy
================

.. c:function:: PetscErrorCode PartitionDestroy(PartitionOfUnity pu, Partition **partition)

   Destroy all partitions in the array.

   **Collective**

   **Input Parameters**

   - ``pu`` (:c:type:`PartitionOfUnity`) - the PartitionOfUnity context
   - ``partition`` (:c:type:`Partition`) - array of Partition structures

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function destroys all partitions in the array by freeing their associated data.

   **See also**

   - :c:func:`PartitionCreate`
   - :c:func:`PartitionDestroy_SinglePartition`

   **Source**

   - ``src/partitions/partition.c``
