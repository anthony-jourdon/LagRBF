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
  file:    test_matmatsolve.c

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

static PetscErrorCode AnalyticalRotationVelocity(PetscReal coor[], PetscReal velocity[])
{
  PetscReal r,theta;
  PetscReal radial_u  = 0.0;
  PetscReal angular_u = 0.5;
  PetscFunctionBeginUser;
  r     = PetscSqrtReal(coor[0]*coor[0] + coor[1]*coor[1]);
  theta = PetscAtan2Real(coor[1], coor[0]);
  velocity[0] = radial_u * PetscCosReal(theta) - angular_u * r * PetscSinReal(theta);
  velocity[1] = radial_u * PetscSinReal(theta) + angular_u * r * PetscCosReal(theta);  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmSetVelocity(DM swarm)
{
  DMSwarmCellDM celldm;
  PetscInt      npoints,dim,p;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  for (p=0; p<npoints; p++) {
    PetscCall(AnalyticalRotationVelocity(&coords[dim*p], &velocity[dim*p]));
  }
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Advect_RK1(DM swarm, PetscReal dt)
{
  DMSwarmCellDM celldm;
  PetscInt      p,dim,npoints,d;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  for (p=0; p<npoints; p++) {
    for (d=0; d<dim; d++) {
      coords[dim*p + d] += dt * velocity[dim*p + d];
    }
  }
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0],NULL, NULL,(void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "velocity",   NULL, NULL,(void **)&velocity));
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE)); // update cell ids and MPI ranks
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  DM            da,swarm;
  PetscInt      dim,nopts,stencil_width,ppcell,step;
  PetscInt      n[] = {5,5};
  PetscReal     dt,time,time_end;
  PetscReal     O[3] = {0.0,0.0,0.0}, L[3] = {1.0,1.0,1.0};
  PetscFunctionBeginUser;
  /* Create DMDA */
  dim = 2;
  nopts = dim;
  stencil_width = 1;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,"-n",n,&nopts,NULL));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,n[0],n[1],PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,&da));
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  nopts = dim;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-O",O,&nopts,NULL));
  nopts = dim;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-L",L,&nopts,NULL));
  PetscCall(DMDASetUniformCoordinates(da,O[0],L[0],O[1],L[1],O[2],L[2]));
  
  /* Create Swarm */
  PetscCall(DMCreate(PetscObjectComm((PetscObject)da), &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, da));
  PetscCall(DMSetFromOptions(swarm));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"velocity",dim,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));
  ppcell = 8;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL));
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm, DMSWARMPIC_LAYOUT_REGULAR, ppcell));
  /* Set velocity */
  PetscCall(SwarmSetVelocity(swarm));
  /* Output */
  PetscCall(DMSwarmViewXDMF(swarm,"test_output/swarm_advection_0.xmf"));
  /* Advect */
  dt = 1e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  step = 1;
  time = dt;
  time_end = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_end",&time_end,NULL));
  while (time < time_end) {
    char fname[PETSC_MAX_PATH_LEN];
    PetscPrintf(PETSC_COMM_WORLD,"Step: %" PetscInt_FMT " Time: %g\n",step,time);
    PetscCall(Advect_RK1(swarm, dt));
    time += dt;
    /* Update velocity */
    PetscCall(SwarmSetVelocity(swarm));
    /* Output */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"test_output/swarm_advection_%" PetscInt_FMT ".xmf",step);
    PetscCall(DMSwarmViewXDMF(swarm,fname));
    step++;
  }
  PetscCall(DMDestroy(&swarm));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}