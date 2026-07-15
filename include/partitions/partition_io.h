#ifndef __LagRBF_partition_io_h
#define __LagRBF_partition_io_h

#include "partitions/partition.h"
#include "io_utils.h"

PetscErrorCode PartitionWriteFactoredMatrix(Partition p, const char *dirname);
PetscErrorCode PartitionLoadFactoredMatrix_SeqDense(Partition p, const char *dirname);

#endif