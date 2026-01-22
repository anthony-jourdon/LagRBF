#include "partitions/partition_io.h"

#include <../src/mat/impls/dense/seq/dense.h>
#include <petscmat.h>

#include "petsc_private/petsc_dense.h"

PetscErrorCode PartitionWriteOriginalMatrix(Partition p, const char *dirname)
{
  PetscViewer       viewer;
  Mat               A;
  PetscMPIInt       rank;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Access the RBF matrix */
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCheck(A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"RBF matrix not created yet. Call PartitionCreateRBFMatrix() first.");
  /* Write the RBF matrix to file */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/MatRBF_r%d_p%"PetscInt_FMT".pbmat",dirname,rank,p->idx));
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(MatView(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionLoadOriginalMatrix(Partition p, const char *dirname)
{
  PetscViewer       viewer;
  Mat               A;
  PetscMPIInt       rank;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&A));
  /* Read the RBF matrix from file */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/MatRBF_r%d_p%"PetscInt_FMT".pbmat",dirname,rank,p->idx));
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  p->A = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionWriteFactoredMatrix(Partition p, const char *dirname)
{
  PetscViewer       viewer;
  IS                is_pivots;
  Mat               A,B;
  Vec               Fwork;
  PetscLayout       rmap,cmap;
  Mat_SeqDense      *mat;
  const PetscScalar *values;
  PetscMPIInt       rank;
  MPI_Comm          Acomm;
  FILE             *fp;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Access the factored RBF matrix */
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PetscObjectGetComm((PetscObject)A,&Acomm));
  /* Create a placeholder matrix */
  PetscCall(MatCreateDense(Acomm,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&B));
  mat = (Mat_SeqDense *)A->data;
  PetscCall(MatDenseGetArrayRead(A,&values));
  /* Fill the placeholder matrix with the values from the factored matrix */
  PetscCall(MatDensePlaceArray(B,values));
  /* Write the placeholder matrix to file */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/MatRBF_r%d_p%"PetscInt_FMT".pbmat",dirname,rank,p->idx));
  PetscCall(PetscViewerCreate(Acomm,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(MatView(B,viewer));
  PetscCall(MatDenseRestoreArrayRead(A,&values));

  /* Write pivots */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Pivots_r%d_p%"PetscInt_FMT".pbis",dirname,rank,p->idx));
  PetscCall(MatGetLayouts(A,&rmap,&cmap));
  PetscCall(ISCreateGeneral(Acomm,rmap->n,mat->pivots,PETSC_USE_POINTER,&is_pivots));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(ISView(is_pivots,viewer));
  PetscCall(ISDestroy(&is_pivots));

  /* Write fwork */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Fwork_r%d_p%"PetscInt_FMT".pbvec",dirname,rank,p->idx));
  PetscCall(VecCreate(Acomm,&Fwork));
  PetscCall(VecSetType(Fwork,VECSEQ));
  PetscCall(VecSetSizes(Fwork,PETSC_DECIDE,mat->lfwork));
  PetscCall(VecPlaceArray(Fwork,mat->fwork));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(VecView(Fwork,viewer));
  PetscCall(VecDestroy(&Fwork));

  /* Write Info */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Info_r%d_p%"PetscInt_FMT".txt",dirname,rank,p->idx));
  fp = fopen(filename,"w");
  if (!fp) { SETERRABORT(Acomm,PETSC_ERR_FILE_OPEN,"Unable to open file %s for writing",filename); }
  if (fp) {
    fprintf(fp,"%"PetscInt_FMT"\n",rmap->n); // matrix rank
    fprintf(fp,"%"PetscInt_FMT"\n",mat->lda); // leading dimension
    fprintf(fp,"%"PetscInt_FMT"\n",mat->lfwork); // length of fwork
    fclose(fp);
  }

  /* Clean-up */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  p->A = NULL; // Mark the RBF matrix as destroyed in the partition structure

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionLoadFactoredMatrix_SeqDense(Partition p, const char *dirname)
{
  PetscViewer       viewer;
  PetscLayout       rmap,cmap;
  Mat_SeqDense      *mat;
  Mat               A;
  Vec               Fwork;
  IS                is_pivots;
  const PetscInt    *pivots;
  PetscInt          mat_rank,lda,lfwork;
  PetscScalar      *fwork;
  PetscMPIInt       rank;
  MPI_Comm          Acomm;
  FILE             *fp;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* First, read the matrix infos */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Info_r%d_p%"PetscInt_FMT".txt",dirname,rank,p->idx));
  fp = fopen(filename,"r");
  if (!fp) { SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file %s for reading",filename); }
  if (fp) {
    fscanf(fp,"%"PetscInt_FMT"\n",&mat_rank);
    fscanf(fp,"%"PetscInt_FMT"\n",&lda);
    fscanf(fp,"%"PetscInt_FMT"\n",&lfwork);
    fclose(fp);
  }
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,p->ndata,p->ndata,NULL,&A));
  PetscCall(PetscObjectGetComm((PetscObject)A,&Acomm));
  mat = (Mat_SeqDense *)A->data;
  /* Create Petsc Viewer */
  PetscCall(PetscViewerCreate(Acomm,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_READ));

  /* Read the factored RBF matrix from file */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/MatRBF_r%d_p%"PetscInt_FMT".pbmat",dirname,rank,p->idx));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(MatLoad(A,viewer));
  /* Set matrix infos */
  PetscCall(MatGetLayouts(A,&rmap,&cmap));
  rmap->n     = mat_rank;
  mat->lda    = lda;
  mat->lfwork = lfwork;
  
  /* Read Pivots */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Pivots_r%d_p%"PetscInt_FMT".pbis",dirname,rank,p->idx));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(ISCreate(Acomm,&is_pivots));
  PetscCall(ISLoad(is_pivots,viewer));
  /* Set the pivots in the matrix */
  PetscCall(ISGetIndices(is_pivots, &pivots));
  if (!mat->pivots) { 
    PetscCall(PetscMalloc1(mat_rank,&mat->pivots));
  } else { 
    PetscCall(PetscFree(mat->pivots));
    PetscCall(PetscMalloc1(mat_rank,&mat->pivots));
  }
  PetscCall(PetscArraycpy(mat->pivots,pivots,mat_rank));
  PetscCall(ISRestoreIndices(is_pivots, &pivots));
  PetscCall(ISDestroy(&is_pivots));

  /* Read fwork */
  PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/Fwork_r%d_p%"PetscInt_FMT".pbvec",dirname,rank,p->idx));
  PetscCall(PetscViewerFileSetName(viewer,filename));
  PetscCall(VecCreate(Acomm,&Fwork));
  PetscCall(VecLoad(Fwork,viewer));
  /* Set fwork in the matrix */
  if (!mat->fwork) { 
    PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
  } else { 
    PetscCall(PetscFree(mat->fwork));
    PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
  }
  PetscCall(VecGetArray(Fwork,&fwork));
  PetscCall(PetscArraycpy(mat->fwork,fwork,mat->lfwork));
  PetscCall(VecRestoreArray(Fwork,&fwork));
  PetscCall(VecDestroy(&Fwork));

  PetscCall(MatSeqDenseAttachCholeskyFunctions(A));

  p->A = A;
  /* Clean-up */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}