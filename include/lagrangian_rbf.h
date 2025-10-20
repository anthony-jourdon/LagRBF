#ifndef __softname_lagrangian_rbf_h
#define __softname_lagrangian_rbf_h

#include <petsc.h>
#include "kernels/kernels_rbf.h"

typedef struct _p_LagRBFCtx        *LagRBFCtx;
typedef struct _p_Partition        *Partition;
typedef struct _p_PartitionOfUnity *PartitionOfUnity;
typedef struct _p_AdvectionCtx     *AdvectionCtx;
typedef struct _p_CellInfo         *CellInfo;

typedef enum {
  GEODYNAMICS = 0, // Geodynamics model
  SURFACE, // Surface process model
  CLIMATE
} InterpModelType;

struct _p_LagRBFCtx {
  PartitionOfUnity pu; // Partition of Unity context
  InterpModelType  mtype; // Model type   
  AdvectionCtx     adv_ctx;  // Advection context
  Partition        *partitions; // Array of partitions structs
  CellInfo         *cellinfo; // Array of cell info structs
  KernelType       kernel_type;
  DM               swd,swi; // Swarm for data points and interpolation points
  PetscInt         n_ifields; // Number of fields to interpolate
  PetscInt         n_pfields; // Number of fields that are not resulting from interpolation
  PetscInt         step; // Current step in the model
  PetscReal        time; // Current time in the model
  PetscBool        restart; // Flag to indicate if the interpolation points coordinates should be loaded from a file
  PetscBool        advection; // Flag to indicate if advection is activated
  char             output_dir[PETSC_MAX_PATH_LEN]; // Directory to write output files
};

PetscErrorCode LagRBFCtxCreate(LagRBFCtx *ctx);
PetscErrorCode LagRBFCtxDestroy(LagRBFCtx *ctx);
PetscErrorCode LagRBF_PartitionOfUnityCreate(LagRBFCtx ctx);
PetscErrorCode LagRBF_PartitionCreate(LagRBFCtx ctx);
PetscErrorCode LagRBF_CellInfoCreate(LagRBFCtx ctx);
PetscErrorCode LagRBFSetFromOptions(LagRBFCtx ctx);
PetscErrorCode LagRBF_AdvectSwarm(LagRBFCtx ctx, DM swi, DM swd);
/* Setters */
PetscErrorCode LagRBFSetKernelType(LagRBFCtx ctx, KernelType type);
PetscErrorCode LagRBFSetStep(LagRBFCtx ctx, PetscInt step);
PetscErrorCode LagRBFSetTimeStep(LagRBFCtx ctx, PetscReal dt);
PetscErrorCode LagRBFSetTime(LagRBFCtx ctx, PetscReal time);
/* Getter */
PetscErrorCode LagRBFGetKernelType(LagRBFCtx ctx, KernelType *type);
PetscErrorCode LagRBFGetPartitionOfUnity(LagRBFCtx ctx, PartitionOfUnity *pu);
PetscErrorCode LagRBFGetPartitions(LagRBFCtx ctx, Partition **partitions);
PetscErrorCode LagRBFGetCellInfo(LagRBFCtx ctx, CellInfo **cellinfo);
PetscErrorCode LagRBFGetStep(LagRBFCtx ctx, PetscInt *step);
PetscErrorCode LagRBFGetTimeStep(LagRBFCtx ctx, PetscReal *dt);
PetscErrorCode LagRBFGetTime(LagRBFCtx ctx, PetscReal *time);
PetscErrorCode LagRBFGetAdvectionCtx(LagRBFCtx ctx, AdvectionCtx *adv_ctx);
PetscErrorCode LagRBFGetSwarmData(LagRBFCtx ctx, DM *swd);
PetscErrorCode LagRBFGetSwarmInterp(LagRBFCtx ctx, DM *swi);
#endif