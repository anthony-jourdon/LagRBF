#ifndef __LagRBF_partition_of_unity_h
#define __LagRBF_partition_of_unity_h

#include <petsc.h>
#include "lagrangian_rbf.h"

struct _p_PartitionOfUnity {
  DM        dap;   // Parallel DMDA mesh COMM_WORLD
  DM        das;   // Sequential DMDA mesh COMM_SELF
  PetscInt  dim;   // Dimension of the partition of unity mesh
  PetscInt  *n;    // Number of partitions in each dimension
  PetscInt  npart; // Rank-local number of partitions
  PetscInt  npg;   // Global number of partitions
  PetscInt  *ptos; // Parallel to sequential mapping of elements
  PetscReal *O,*L; // Origin and Length of the partition of unity mesh
  PetscReal epsilon_w; // Support factor for the weights RBF kernel
  PetscReal epsilon_r; // Shape parameter for the interpolation RBF kernel
  PetscReal overlap; // Overlap factor for the partition of unity
  PetscReal shape_fac; // Factor to compute the shape parameter as epsilon_r = shape_fac * epsilon_w
  char      name[PETSC_MAX_PATH_LEN]; // Name of the parallel DMDA mesh
  char      sname[PETSC_MAX_PATH_LEN]; // Name of the sequential DMDA mesh
  
  // Partitions methods
  PetscBool      psetup_set;
  PetscErrorCode (*psetup)(LagRBFCtx,Partition,PetscReal*);
  PetscErrorCode (*pdestroy)(Partition);
  PetscErrorCode (*psolve)(LagRBFCtx,Partition,PetscInt,PetscReal*);
};

PetscErrorCode PUCreate(PartitionOfUnity *pu);
PetscErrorCode PUCreateDMDA(PartitionOfUnity pu);
PetscErrorCode PUCreateDMDALocal(PartitionOfUnity pu);
PetscErrorCode PartitionOfUnityCreate(PartitionOfUnity *pu);
PetscErrorCode PartitionOfUnityDestroy(PartitionOfUnity *pu);
PetscErrorCode PartitionOfUnityComputeSupport(PartitionOfUnity pu);
PetscErrorCode PartitionOfUnityComputeShapeFactor(PartitionOfUnity pu);
PetscErrorCode PUAttachPartitionFunctions(PartitionOfUnity pu, ASolveMethod smethod);
/* Getters */
PetscErrorCode PUGetDMDA(PartitionOfUnity pu, DM *dap);
PetscErrorCode PUGetDMDALocal(PartitionOfUnity pu, DM *das);
PetscErrorCode PUGetNPartitionsLocal(PartitionOfUnity pu, PetscInt *npart);
PetscErrorCode PUGetNPartitionsGlobal(PartitionOfUnity pu, PetscInt *npg);
PetscErrorCode PUGetDimension(PartitionOfUnity pu, PetscInt *dim);
PetscErrorCode PUGetGlobalToLocalElementMap(PartitionOfUnity pu, PetscInt **ptos);
PetscErrorCode PUGetSupportFactor(PartitionOfUnity pu, PetscReal *epsilon);
PetscErrorCode PUGetShapeFactor(PartitionOfUnity pu, PetscReal *epsilon);
/* Output */
PetscErrorCode PartitionOfUnityMeshView(PartitionOfUnity pu);
PetscErrorCode PartitionOfUnityMeshView_Seq(PartitionOfUnity pu);
#endif