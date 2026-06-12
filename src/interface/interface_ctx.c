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

void lagrbf_destroy(void *ctx)
{
  LagRBFCtx p = (LagRBFCtx)ctx;
  if (p) { LagRBFCtxDestroy(&p); }
  return;
}