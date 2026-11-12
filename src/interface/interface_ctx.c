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
  file:    interface_ctx.c

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
#include "lagrangian_rbf_init.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_data.h"
#include "cell_info.h"
#include "partitions/partition.h"
#include "interpolation.h"
#include "interface/interface_ctx.h"
#include "interface/interface_utils.h"

/*@C
  Interface_LagRBFCreate - Creates the LagRBFCtx context and initializes the data structures from options.

  Collective
  Input Parameters:
. ctx (LagRBFCtx*) - pointer to the LagRBFCtx context to be created

  Output Parameter:
. PETSC_SUCCESS - if successful

  Notes:
  This function initializes the data structures in the memory. User must call :c:func:`LagRBFCtxDestroy()` to free the memory.

.seealso: LagRBFCtxDestroy(), lagrbf_create(), lagrbf_destroy(), LagRBFCtxCreate(), LagRBFSetFromOptions(), LagRBF_PartitionOfUnityCreate(), LagRBF_PartitionCreate(), LagRBF_CellInfoCreate(), SwarmDataCreate(), SwarmInterpCreate()
@*/
PetscErrorCode Interface_LagRBFCreate(LagRBFCtx *ctx)
{
  LagRBFCtx user;
  DM        swarm_data, swarm_interp;
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(0,NULL,NULL,NULL));

  /* Create context */
  PetscCall(LagRBFCtxCreate(&user));
  PetscCall(LagRBFSetFromOptions(user));
  /* Create data structures */
  PetscCall(LagRBF_PartitionOfUnityCreate(user));
  PetscCall(LagRBF_PartitionCreate(user));
  PetscCall(LagRBF_CellInfoCreate(user));
  /* Create the swarm to hold data */
  PetscCall(SwarmDataCreate(user,&swarm_data));
  user->swd = swarm_data;
  /* Create the swarm to interpolate on */
  PetscCall(SwarmInterpCreate(user,&swarm_interp));
  user->swi = swarm_interp;

  *ctx = user;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  lagrbf_create - Creates the LagRBFCtx context and initializes the data structures from options. To be called from Python.

  Collective

  Output Parameter:
. ctx (void*) - pointer to the LagRBFCtx context to be created

  Notes:
  This function initializes the data structures in the memory. User must call :c:func:`lagrbf_destroy` to free the memory.
  This function is a wrapper for :c:func:`Interface_LagRBFCreate` to be called from Python.

.seealso: lagrbf_destroy(), Interface_LagRBFCreate()
@*/
void *lagrbf_create()
{
  LagRBFCtx ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = Interface_LagRBFCreate(&ctx); 
  if (ierr) { 
    InterfacePetscErrorPrinter(ierr, "lagrbf_create");
    PetscFunctionReturn(NULL); 
  }
  PetscFunctionReturn((void *)ctx);
}

/*@C
  lagrbf_destroy - Destroys the LagRBFCtx context and frees the memory.

  Collective

  Input Parameter:
. ctx (void*) - pointer to the LagRBFCtx context to be destroyed

  Notes:
  This function frees the memory allocated for the LagRBFCtx context. User must call this function when they are done with the context.
  This function is a wrapper for :c:func:`LagRBFCtxDestroy` to be called from Python.

.seealso: lagrbf_create(), LagRBFCtxDestroy(), Interface_LagRBFCreate()
@*/
void lagrbf_destroy(void *ctx)
{
  LagRBFCtx p = (LagRBFCtx)ctx;
  if (p) { LagRBFCtxDestroy(&p); }
  return;
}