#include <petsc.h>
#include "kernels/kernels_rbf.h"

PetscReal KernelLinear(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  PetscFunctionReturn(u);
}

PetscReal KernelWendlandC0(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  if (u < 0.0) PetscFunctionReturn(0.0);
  PetscFunctionReturn(u*u);
}

PetscReal KernelWendlandC2(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  if (u < 0.0) PetscFunctionReturn(0.0);
  PetscFunctionReturn(pow(u, 4) * (5.0 - 4.0 * u));
}

PetscReal KernelWendlandC4(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  if (u < 0.0) PetscFunctionReturn(0.0);
  PetscFunctionReturn(pow(u, 6) * (56.0 - 88.0 * u + 35.0 * u * u));
}

PetscReal KernelWendlandC6(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  if (u < 0.0) PetscFunctionReturn(0.0);
  PetscFunctionReturn(pow(u, 8) * (66.0 - 154.0 * u + 121.0 * u * u - 32.2 * u * u * u));
}

PetscErrorCode KernelGetFunction(KernelType ktype, PetscReal (**kernel_fn)(PetscReal,PetscReal))
{
  PetscFunctionBeginUser;
  switch (ktype) {
    case KERNEL_LINEAR:
      *kernel_fn = KernelLinear;
      break;
    case KERNEL_WENDLAND_C0:
      *kernel_fn = KernelWendlandC0;
      break;
    case KERNEL_WENDLAND_C2:
      *kernel_fn = KernelWendlandC2;
      break;
    case KERNEL_WENDLAND_C4:
      *kernel_fn = KernelWendlandC4;
      break;
    case KERNEL_WENDLAND_C6:
      *kernel_fn = KernelWendlandC6;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown kernel type %d",ktype);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(__AVX__)
#include "kernels/kernels_rbf_avx.h"
PetscErrorCode KernelGetFunction_AVX256(KernelType ktype, __m256d (**kernel_avx)(__m256d,__m256d))
{
  PetscFunctionBeginUser;
  switch (ktype) {
    case KERNEL_LINEAR:
      *kernel_avx = KernelLinear_AVX256;
      break;
    case KERNEL_WENDLAND_C2:
      *kernel_avx = KernelWendlandC2_AVX256;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown kernel type %d for AVX256",ktype);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if defined(__AVX512F__)
#include "kernels/kernels_rbf_avx512.h"
PetscErrorCode KernelGetFunction_AVX512(KernelType ktype, __m512d (**kernel_avx)(__m512d,__m512d))
{
  PetscFunctionBeginUser;
  switch (ktype) {
    case KERNEL_LINEAR:
      *kernel_avx = KernelLinear_AVX512;
      break;
    case KERNEL_WENDLAND_C2:
      *kernel_avx = KernelWendlandC2_AVX512;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown kernel type %d for AVX512",ktype);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif