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
  file:    runge_kutta_2.c

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

#include <petsc.h>
#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "advection/advection.h"
#include "advection/runge_kutta_2.h"
#include "swarm/swarm_interp.h"
#include "interpolation.h"

PetscLogEvent LAGRBF_AdvectionExplicit_RK2;

PetscErrorCode AdvectionExplicit_RK2(LagRBFCtx ctx, DM swi, DM swd)
{
  DMSwarmCellDM celldm;
  AdvectionCtx  adv;
  PetscInt      dim,npoints;
  PetscReal     dt;
  PetscReal     *x0,*coords,*velocity;
  const char    *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_AdvectionExplicit_RK2,0,0,0,0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME));

  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv));
  PetscCall(AdvectionGetTimeStep(adv,&dt));
  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(adv, &velocity_name));
  /* Access fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  /* Store initial position x0 */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* Step 1: compute x1 = x0 + 0.5*dt*v0 */
  PetscCall(AdvectionExplicit(coords,x0,velocity,0.5*dt,npoints,dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation
  /* Step 2: Evaluate v1 = v(x1) */
  PetscCall(AdvectionEvaluateVelocity(ctx,swi,swd));
  /* Access fields */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  /* Step 3: Advect x = x0 + dt*v1 */
  PetscCall(AdvectionExplicit(coords,x0,velocity,dt,npoints,dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  /* Update points */
  PetscCall(DMSwarmMigrate(swi,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscCall(SwarmInterpSetWeights(ctx,swi)); // update the weights for the interpolation

  PetscCall(PetscLogEventEnd(LAGRBF_AdvectionExplicit_RK2,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Advection_RK2Create(AdvectionCtx adv_ctx)
{
  PetscFunctionBeginUser;
  adv_ctx->advection = AdvectionExplicit_RK2;
  adv_ctx->destroy   = NULL;
  adv_ctx->order     = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}