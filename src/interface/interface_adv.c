/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    interface_adv.c

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "time_loop.h"
#include "interface/interface_utils.h"
#include "interface/interface_adv.h"
#include "advection/advection.h"

PetscErrorCode lagrbf_advection_function(
  void *ctx,
  PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), 
  void *fn_ctx
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  AdvectionCtx   adv_ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(user,&adv_ctx));
  ierr = AdvectionSetVelocityFunction(adv_ctx, velocity_fn, fn_ctx);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "AdvectionSetVelocityFunction");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_advection_function_batch(
  void *ctx,
  PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), 
  void *fn_ctx
)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  AdvectionCtx   adv_ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetAdvectionCtx(user,&adv_ctx));
  ierr = AdvectionSetVelocityFunctionBatch(adv_ctx, velocity_fn_batch, fn_ctx);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "AdvectionSetVelocityFunctionBatch");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_avect_analytical(void *ctx, PetscReal time, PetscReal time_next)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  PetscCheck(user,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx is NULL");
  PetscCheck(user->swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Interpolation swarm (ctx->swi) is NULL");
  ierr = ForwardAnalyticalAdvection(user, user->swi, time, time_next);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "ForwardAnalyticalAdvection");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lagrbf_advect_interpolate(void *ctx, PetscReal time, PetscReal time_next)
{
  LagRBFCtx      user = (LagRBFCtx)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscCheck(user,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "LagRBFCtx is NULL");
  PetscCheck(user->swi, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Interpolation swarm (ctx->swi) is NULL");
  PetscCheck(user->swd, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Data swarm (ctx->swd) is NULL");

  ierr = InternalForwardAdvectionLoop(user,user->swd,user->swi,time,time_next);
  if (ierr) {
    InterfacePetscErrorPrinter(ierr, "InternalForwardAdvectionLoop");
    PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}