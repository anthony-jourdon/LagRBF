#ifndef __softname_lagrangian_rbf_h
#define __softname_lagrangian_rbf_h

#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"

typedef enum {
  GEODYNAMICS = 0, // Geodynamics model
  SURFACE, // Surface process model
  CLIMATE
} InterpModelType;

struct _p_LagRBFCtx {
  PartitionOfUnity pu; // Partition of Unity context
  InterpModelType  mtype; // Model type   
  Partition        *partitions; // Array of partitions structs
  PetscInt         n_ifields; // Number of fields to interpolate
  PetscInt         n_pfields; // Number of fields that are not resulting from interpolation
  PetscInt         int_order; // Integration order for advection scheme
  PetscInt         step; // Current step in the model
  PetscReal        time; // Current time in the model
  PetscReal        dt; // Time step size
  PetscBool        restart; // Flag to indicate if the interpolation points coordinates should be loaded from a file
  char             output_dir[PETSC_MAX_PATH_LEN]; // Directory to write output files
};

typedef struct _p_LagRBFCtx *LagRBFCtx;

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx);
PetscErrorCode LagRBFCtxDestroy(LagRBFCtx *ctx);
PetscErrorCode LagRBF_PartitionOfUnityCreate(LagRBFCtx ctx);
PetscErrorCode LagRBF_PartitionCreate(LagRBFCtx ctx);
PetscErrorCode LagRBFSetFromOptions(LagRBFCtx ctx);
/* Setters */
PetscErrorCode LagRBFSetStep(LagRBFCtx ctx, PetscInt step);
PetscErrorCode LagRBFSetTimeStep(LagRBFCtx ctx, PetscReal dt);
PetscErrorCode LagRBFSetTime(LagRBFCtx ctx, PetscReal time);
/* Getter */
PetscErrorCode LagRBFGetPartitionOfUnity(LagRBFCtx ctx, PartitionOfUnity *pu);
PetscErrorCode LagRBFGetPartitions(LagRBFCtx ctx, Partition **partitions);
PetscErrorCode LagRBFGetStep(LagRBFCtx ctx, PetscInt *step);
PetscErrorCode LagRBFGetTimeStep(LagRBFCtx ctx, PetscReal *dt);
PetscErrorCode LagRBFGetTime(LagRBFCtx ctx, PetscReal *time);
#endif