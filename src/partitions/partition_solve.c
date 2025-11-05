#include <petsc.h>

#include "partition_of_unity.h"
#include "partitions/partition.h"
#include "kernels/kernels_rbf.h"
#include "partitions/partition_solve.h"

PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix;
PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;

PetscErrorCode PartitionCreateRBFMatrix(Partition p)
{
  PetscFunctionBeginUser;
  PetscCheck(p,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition structure is NULL. Call PartitionCreate() first.");
  PetscCheck(!p->A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix already created.");
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&p->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionAssembleRBFMatrix(
  Partition p, 
  PetscInt dim, 
  PetscReal coords[], 
  PetscReal (*kernel_fn)(PetscReal,PetscReal)
)
{
  Mat       A;
  PetscInt  i,j,d,npoints;
  PetscInt  *pidx;
  PetscReal epsilon;
  PetscReal *rbf;
  PetscFunctionBeginUser;

  PetscCall(PetscLogEventBegin(LAGRBF_PartitionAssembleRBFMatrix,0,0,0,0));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
  PetscCall(PartitionGetEpsilon(p,&epsilon));
  PetscCall(MatDenseGetArray(A,&rbf));
  /* Fill the matrix */
  for (i=0; i<npoints; i++) {
    for (j=i; j<npoints; j++) {
      PetscReal dist = 0.0;
      /* Compute the distance between points */
      for (d=0; d<dim; d++) {
        dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
      }
      dist = PetscSqrtReal(dist);
      /* Fill the upper triangular part with the RBF kernel */
      rbf[npoints*i + j] = kernel_fn(dist,epsilon);
      /* Copy the value to the lower triangular part (diagonal already filled) */
      if (i != j) {
        rbf[npoints*j + i] = rbf[npoints*i + j];
      }
    }
  }
  PetscCall(MatDenseRestoreArray(A,&rbf));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionAssembleRBFMatrix,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionSolveInterpolationCoefficients(PartitionOfUnity pu, Partition *partitions, DM swarm, const char *fieldname)
{
  PetscInt  n,npart,nfields;
  PetscReal *values;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_PartitionSolveInterpolationCoefficients,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscCall(DMSwarmGetField(swarm,fieldname,&nfields,NULL,(void **)&values));
  for (n=0; n<npart; n++) {
    Partition p = partitions[n];
    PetscInt  npoints;
    PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
    if (npoints == 0) continue; // no data points in this partition, skip
    /* Solve for the interpolation coefficients in this partition */
    PetscCall(p->solve(p,nfields,values));
  }
  /* Restore the field */
  PetscCall(DMSwarmRestoreField(swarm,fieldname,&nfields,NULL,(void **)&values));
  PetscCall(PetscLogEventEnd(LAGRBF_PartitionSolveInterpolationCoefficients,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}