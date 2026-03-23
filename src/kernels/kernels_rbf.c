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

PetscReal KernelBump(PetscReal r, PetscReal epsilon)
{
  PetscReal u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  if (u < 0.0) PetscFunctionReturn(0.0);
  return exp(-1.0 / (u * u));
}

PetscReal KernelMultiquadric(PetscReal r, PetscReal epsilon)
{
  PetscReal rbf,u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  rbf = sqrt(1.0 + u*u);
  PetscFunctionReturn(rbf);
}

PetscReal KernelInverseMultiquadric(PetscReal r, PetscReal epsilon)
{
  PetscReal rbf,u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  rbf = 1.0 / sqrt(1.0 + u*u);
  PetscFunctionReturn(rbf);
}

PetscReal KernelGaussian(PetscReal r, PetscReal epsilon)
{
  PetscReal rbf,u;
  PetscFunctionBeginUser;
  u = 1.0 - epsilon * r;
  rbf = exp(-u*u);
  PetscFunctionReturn(rbf);
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
    case KERNEL_BUMP:
      *kernel_fn = KernelBump;
      break;
    case KERNEL_MQ:
      *kernel_fn = KernelMultiquadric;
      break;
    case KERNEL_IMQ:
      *kernel_fn = KernelInverseMultiquadric;
      break;
    case KERNEL_GAUSSIAN:
      *kernel_fn = KernelGaussian;
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
    case KERNEL_WENDLAND_C0:
      *kernel_avx = KernelWendlandC0_AVX256;
      break;
    case KERNEL_WENDLAND_C2:
      *kernel_avx = KernelWendlandC2_AVX256;
      break;
    case KERNEL_BUMP:
      *kernel_avx = KernelBump_AVX256;
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
    case KERNEL_WENDLAND_C0:
      *kernel_avx = KernelWendlandC0_AVX512;
      break;
    case KERNEL_WENDLAND_C2:
      *kernel_avx = KernelWendlandC2_AVX512;
      break;
    case KERNEL_BUMP:
      *kernel_avx = KernelBump_AVX512;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown kernel type %d for AVX512",ktype);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif