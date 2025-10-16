#ifndef __noname_partition_of_unity_h
#define __noname_partition_of_unity_h

#include <petsc.h>

struct _p_PartitionOfUnity {
  DM        dap;   // Parallel DMDA mesh COMM_WORLD
  DM        das;   // Sequential DMDA mesh COMM_SELF
  PetscInt  dim;   // Dimension of the partition of unity mesh
  PetscInt  *n;    // Number of partitions in each dimension
  PetscInt  npart; // Rank-local number of partitions
  PetscInt  npg;   // Global number of partitions
  PetscInt  *ptos; // Parallel to sequential mapping of elements
  PetscReal *O,*L; // Origin and Length of the partition of unity mesh
  PetscReal epsilon; // Epsilon factor for the RBF kernel
  char      name[PETSC_MAX_PATH_LEN]; // Name of the parallel DMDA mesh
  char      sname[PETSC_MAX_PATH_LEN]; // Name of the sequential DMDA mesh
};

typedef struct _p_PartitionOfUnity *PartitionOfUnity;

PetscErrorCode PUCreate(PartitionOfUnity *pu);
PetscErrorCode PUCreateDMDA(PartitionOfUnity pu);
PetscErrorCode PUCreateDMDALocal(PartitionOfUnity pu);
PetscErrorCode PartitionOfUnityCreate(PartitionOfUnity *pu);
PetscErrorCode PartitionOfUnityDestroy(PartitionOfUnity *pu);
PetscErrorCode PartitionOfUnityComputeEpsilon(PartitionOfUnity pu);
/* Getters */
PetscErrorCode PUGetDMDA(PartitionOfUnity pu, DM *dap);
PetscErrorCode PUGetDMDALocal(PartitionOfUnity pu, DM *das);
PetscErrorCode PUGetNPartitionsLocal(PartitionOfUnity pu, PetscInt *npart);
PetscErrorCode PUGetNPartitionsGlobal(PartitionOfUnity pu, PetscInt *npg);
PetscErrorCode PUGetDimension(PartitionOfUnity pu, PetscInt *dim);
PetscErrorCode PUGetGlobalToLocalElementMap(PartitionOfUnity pu, PetscInt **ptos);
PetscErrorCode PUGetEpsilon(PartitionOfUnity pu, PetscReal *epsilon);
/* Output */
PetscErrorCode PartitionOfUnityMeshView(PartitionOfUnity pu);
PetscErrorCode PartitionOfUnityMeshView_Seq(PartitionOfUnity pu);
#endif