/*
     Defines the basic matrix operations for sequential dense.
     Portions of this code are under:
     Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/mpi/mpidense.h>
#include <petscblaslapack.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/vecimpl.h>

#include "petsc_private/petsc_dense.h"

static PetscErrorCode MatSolve_SeqDense_Internal_Cholesky(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k, PetscBool T)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;
  PetscBLASInt  info;

  PetscFunctionBegin;
  if (A->spd == PETSC_BOOL3_TRUE) {
    if (PetscDefined(USE_COMPLEX) && T) PetscCall(MatConjugate_SeqDense(A));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscCallBLAS("LAPACKpotrs", LAPACKpotrs_("L", &m, &nrhs, mat->v, &mat->lda, x, &m, &info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "POTRS Bad solve %" PetscBLASInt_FMT, info);
    if (PetscDefined(USE_COMPLEX) && T) PetscCall(MatConjugate_SeqDense(A));
#if defined(PETSC_USE_COMPLEX)
  } else if (A->hermitian == PETSC_BOOL3_TRUE) {
    if (T) PetscCall(MatConjugate_SeqDense(A));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscCallBLAS("LAPACKhetrs", LAPACKhetrs_("L", &m, &nrhs, mat->v, &mat->lda, mat->pivots, x, &m, &info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "HETRS Bad solve %" PetscBLASInt_FMT, info);
    if (T) PetscCall(MatConjugate_SeqDense(A));
#endif
  } else { /* symmetric case */
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscCallBLAS("LAPACKsytrs", LAPACKsytrs_("L", &m, &nrhs, mat->v, &mat->lda, mat->pivots, x, &m, &info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "SYTRS Bad solve %" PetscBLASInt_FMT, info);
  }
  PetscCall(PetscLogFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqDense_SetUp(Mat A, Vec xx, Vec yy, PetscScalar **_y, PetscBLASInt *_m, PetscBLASInt *_k)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;
  PetscScalar  *y;
  PetscBLASInt  m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscBLASIntCast(A->cmap->n, &k));
  if (k < m) {
    PetscCall(VecCopy(xx, mat->qrrhs));
    PetscCall(VecGetArray(mat->qrrhs, &y));
  } else {
    PetscCall(VecCopy(xx, yy));
    PetscCall(VecGetArray(yy, &y));
  }
  *_y = y;
  *_k = k;
  *_m = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqDense_TearDown(Mat A, Vec xx, Vec yy, PetscScalar **_y, PetscBLASInt *_m, PetscBLASInt *_k)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;
  PetscScalar  *y   = NULL;
  PetscBLASInt  m, k;

  PetscFunctionBegin;
  y   = *_y;
  *_y = NULL;
  k   = *_k;
  m   = *_m;
  if (k < m) {
    PetscScalar *yv;
    PetscCall(VecGetArray(yy, &yv));
    PetscCall(PetscArraycpy(yv, y, k));
    PetscCall(VecRestoreArray(yy, &yv));
    PetscCall(VecRestoreArray(mat->qrrhs, &y));
  } else {
    PetscCall(VecRestoreArray(yy, &y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar *y = NULL;
  PetscBLASInt m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_FALSE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar *y = NULL;
  PetscBLASInt m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_TRUE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_SeqDense_SetUp(Mat A, Mat B, Mat X, PetscScalar **_y, PetscBLASInt *_ldy, PetscBLASInt *_m, PetscBLASInt *_nrhs, PetscBLASInt *_k)
{
  const PetscScalar *b;
  PetscScalar       *y;
  PetscInt           n, _ldb, _ldx;
  PetscBLASInt       nrhs = 0, m = 0, k = 0, ldb = 0, ldx = 0, ldy = 0;

  PetscFunctionBegin;
  *_ldy  = 0;
  *_m    = 0;
  *_nrhs = 0;
  *_k    = 0;
  *_y    = NULL;
  PetscCall(PetscBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscBLASIntCast(A->cmap->n, &k));
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(PetscBLASIntCast(n, &nrhs));
  PetscCall(MatDenseGetLDA(B, &_ldb));
  PetscCall(PetscBLASIntCast(_ldb, &ldb));
  PetscCall(MatDenseGetLDA(X, &_ldx));
  PetscCall(PetscBLASIntCast(_ldx, &ldx));
  if (ldx < m) {
    PetscCall(MatDenseGetArrayRead(B, &b));
    PetscCall(PetscMalloc1(nrhs * m, &y));
    if (ldb == m) {
      PetscCall(PetscArraycpy(y, b, ldb * nrhs));
    } else {
      for (PetscInt j = 0; j < nrhs; j++) PetscCall(PetscArraycpy(&y[j * m], &b[j * ldb], m));
    }
    ldy = m;
    PetscCall(MatDenseRestoreArrayRead(B, &b));
  } else {
    if (ldb == ldx) {
      PetscCall(MatCopy(B, X, SAME_NONZERO_PATTERN));
      PetscCall(MatDenseGetArray(X, &y));
    } else {
      PetscCall(MatDenseGetArray(X, &y));
      PetscCall(MatDenseGetArrayRead(B, &b));
      for (PetscInt j = 0; j < nrhs; j++) PetscCall(PetscArraycpy(&y[j * ldx], &b[j * ldb], m));
      PetscCall(MatDenseRestoreArrayRead(B, &b));
    }
    ldy = ldx;
  }
  *_y    = y;
  *_ldy  = ldy;
  *_k    = k;
  *_m    = m;
  *_nrhs = nrhs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_SeqDense_TearDown(Mat A, Mat B, Mat X, PetscScalar **_y, PetscBLASInt *_ldy, PetscBLASInt *_m, PetscBLASInt *_nrhs, PetscBLASInt *_k)
{
  PetscScalar *y;
  PetscInt     _ldx;
  PetscBLASInt k, ldy, nrhs, ldx = 0;

  PetscFunctionBegin;
  y    = *_y;
  *_y  = NULL;
  k    = *_k;
  ldy  = *_ldy;
  nrhs = *_nrhs;
  PetscCall(MatDenseGetLDA(X, &_ldx));
  PetscCall(PetscBLASIntCast(_ldx, &ldx));
  if (ldx != ldy) {
    PetscScalar *xv;
    PetscCall(MatDenseGetArray(X, &xv));
    for (PetscInt j = 0; j < nrhs; j++) PetscCall(PetscArraycpy(&xv[j * ldx], &y[j * ldy], k));
    PetscCall(MatDenseRestoreArray(X, &xv));
    PetscCall(PetscFree(y));
  } else {
    PetscCall(MatDenseRestoreArray(X, &y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar *y;
  PetscBLASInt m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_FALSE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar *y;
  PetscBLASInt m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_TRUE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseAttachCholeskyFunctions(Mat A)
{
  PetscFunctionBegin;
  A->ops->solve             = MatSolve_SeqDense_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDense_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_Cholesky;
  A->factortype             = MAT_FACTOR_CHOLESKY;
  PetscFunctionReturn(PETSC_SUCCESS);
}