#include <petsc.h>

PetscReal KernelLinear(PetscReal r, PetscReal epsilon)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(r);
}

PetscReal KernelGaussian(PetscReal r, PetscReal epsilon)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PetscExpReal(-epsilon * epsilon * r * r));
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