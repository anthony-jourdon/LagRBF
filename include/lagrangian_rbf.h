#ifndef __softname_lagrangian_rbf_h
#define __softname_lagrangian_rbf_h

#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"

struct _p_LagRBFCtx {
  PartitionOfUnity pu; // Partition of Unity context
  Partition        *partitions; // Array of partitions structs
  PetscInt         n_ifields; // Number of fields to interpolate
  PetscInt         n_pfields; // Number of fields that are not resulting from interpolation
  PetscBool        restart; // Flag to indicate if the interpolation points coordinates should be loaded from a file
};

typedef struct _p_LagRBFCtx *LagRBFCtx;

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx);
PetscErrorCode LagRBFCtxDestroy(LagRBFCtx *ctx);
PetscErrorCode LagRBF_PartitionOfUnityCreate(LagRBFCtx ctx);
PetscErrorCode LagRBF_PartitionCreate(LagRBFCtx ctx);
PetscErrorCode LagRBFSetFromOptions(LagRBFCtx ctx);
PetscErrorCode LagRBFGetPartitionOfUnity(LagRBFCtx ctx, PartitionOfUnity *pu);
PetscErrorCode LagRBFGetPartitions(LagRBFCtx ctx, Partition **partitions);

#endif