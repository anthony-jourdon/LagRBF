#ifndef __softname_interpolation_h
#define __softname_interpolation_h

#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"

PetscErrorCode RBFInterpolation(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd);

#endif