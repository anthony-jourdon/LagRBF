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
  file:    lagrangian_rbf_init.c

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

PetscClassId LAGRBF_CLASSID;
extern PetscLogEvent LAGRBF_SwarmInterpSetWeights_Scalar;
extern PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX256;
extern PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX512;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_AVX512;
extern PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_Scalar;
extern PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_AVX256;
extern PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix_AVX512;
extern PetscLogEvent LAGRBF_FactorRBFMatrix;
extern PetscLogEvent LAGRBF_MatSetRHS_Fac;
extern PetscLogEvent LAGRBF_MatSetRHS_KSP;
extern PetscLogEvent LAGRBF_SolveLocalCoefficients_Fac;
extern PetscLogEvent LAGRBF_SolveLocalCoefficients_KSP;
extern PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;
extern PetscLogEvent LAGRBF_AdvectionExplicit_scalar;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK1;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK2;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK4;
extern PetscLogEvent LAGRBF_AdvectionExplicit_DOPRI;
extern PetscLogEvent LAGRBF_SphericalCoordinates_Halton;
extern PetscLogEvent LAGRBF_CellInfoSetUp;

static PetscErrorCode LagRBFWriteInitialInfo(void)
{
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██                               ░█████████  ░████████   ░██████████\n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██                               ░██     ░██ ░██    ░██  ░██        \n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██          ░██████    ░████████ ░██     ░██ ░██    ░██  ░██        \n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██               ░██  ░██    ░██ ░█████████  ░████████   ░█████████ \n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██          ░███████  ░██    ░██ ░██   ░██   ░██     ░██ ░██        \n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██         ░██   ░██  ░██   ░███ ░██    ░██  ░██     ░██ ░██        \n");
  PetscPrintf(PETSC_COMM_WORLD,"  ░██████████  ░█████░██  ░█████░██ ░██     ░██ ░█████████  ░██        \n");
  PetscPrintf(PETSC_COMM_WORLD,"                                ░██                                    \n");
  PetscPrintf(PETSC_COMM_WORLD,"                          ░███████                                     \n");
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  PetscPrintf(PETSC_COMM_WORLD,"———————————————————————————————————————————————————————————————————————\n");
  PetscPrintf(PETSC_COMM_WORLD," LagRBF: Lagrangian points Radial Basis Function library\n");
  PetscPrintf(PETSC_COMM_WORLD,"    git url: https://github.com/anthony-jourdon/LagRBF\n");
#if defined(__AVX512F__) 
  PetscPrintf(PETSC_COMM_WORLD,"** AVX-512 detected - optimized kernels will be used.\n");
#elif defined(__AVX__)
  PetscPrintf(PETSC_COMM_WORLD,"** AVX detected - optimized kernels will be used.\n");
#else
  PetscPrintf(PETSC_COMM_WORLD,"** AVX not detected - scalar kernels will be used.\n");
  PetscPrintf(PETSC_COMM_WORLD,"** If your system supports AVX, consider adding options,\n");
  PetscPrintf(PETSC_COMM_WORLD,"** e.g. -march=native, to LAG_CFLAGS in makefile.arch\n");
#endif
  PetscPrintf(PETSC_COMM_WORLD,"———————————————————————————————————————————————————————————————————————\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscBool initialized = PETSC_FALSE;
  PetscFunctionBeginUser;

  /* Check if petsc has already been initialized */
  PetscCall(PetscInitialized(&initialized));
  /* Initialize petsc if not already done */
  if (!initialized) { PetscCall(PetscInitialize(argc,args,file,help)); }
  /* If already initialized and a file is provided, insert the file to options */
  else if (file)    { PetscCall(PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL,file,PETSC_TRUE)); }

  PetscCall(PetscLogDefaultBegin());
  PetscCall(LagRBFWriteInitialInfo());

  /* Register the class */
  PetscCall(PetscClassIdRegister("Lagrangian RBF", &LAGRBF_CLASSID));

  /* Interpolation */
  PetscCall(PetscLogEventRegister("SwarmInterpSetWeights_Scalar",LAGRBF_CLASSID,&LAGRBF_SwarmInterpSetWeights_Scalar));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_Scalar",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_Scalar));
  /* Coefficients solve */
  PetscCall(PetscLogEventRegister("PartitionAssembleRBFMatrix_Scalar",LAGRBF_CLASSID,&LAGRBF_PartitionAssembleRBFMatrix_Scalar));
  PetscCall(PetscLogEventRegister("FactorRBFMatrix",LAGRBF_CLASSID,&LAGRBF_FactorRBFMatrix));
  PetscCall(PetscLogEventRegister("MatSetRHS_Fac",LAGRBF_CLASSID,&LAGRBF_MatSetRHS_Fac));
  PetscCall(PetscLogEventRegister("MatSetRHS_KSP",LAGRBF_CLASSID,&LAGRBF_MatSetRHS_KSP));
  PetscCall(PetscLogEventRegister("SolveLocalCoefficients_Fac",LAGRBF_CLASSID,&LAGRBF_SolveLocalCoefficients_Fac));
  PetscCall(PetscLogEventRegister("SolveLocalCoefficients_KSP",LAGRBF_CLASSID,&LAGRBF_SolveLocalCoefficients_KSP));
  PetscCall(PetscLogEventRegister("PartitionSolveInterpolationCoefficients",LAGRBF_CLASSID,&LAGRBF_PartitionSolveInterpolationCoefficients));
  /* Advection */
  PetscCall(PetscLogEventRegister("AdvectionExplicit_scalar",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_scalar));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK1",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK1));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK2",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK2));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK4",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK4));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_DOPRI",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_DOPRI));

#if defined (__AVX__)
  PetscCall(PetscLogEventRegister("SwarmInterpSetWeights_AVX256",LAGRBF_CLASSID,&LAGRBF_SwarmInterpSetWeights_AVX256));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_AVX256",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_AVX256));
  PetscCall(PetscLogEventRegister("PartitionAssembleRBFMatrix_AVX256",LAGRBF_CLASSID,&LAGRBF_PartitionAssembleRBFMatrix_AVX256));
#endif
#if defined (__AVX512F__)
  PetscCall(PetscLogEventRegister("SwarmInterpSetWeights_AVX512",LAGRBF_CLASSID,&LAGRBF_SwarmInterpSetWeights_AVX512));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_AVX512",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_AVX512));
  PetscCall(PetscLogEventRegister("PartitionAssembleRBFMatrix_AVX512",LAGRBF_CLASSID,&LAGRBF_PartitionAssembleRBFMatrix_AVX512));
#endif
  /* Utilities */
  PetscCall(PetscLogEventRegister("SphericalCoordinates_Halton",LAGRBF_CLASSID,&LAGRBF_SphericalCoordinates_Halton));
  PetscCall(PetscLogEventRegister("CellInfoSetUp",LAGRBF_CLASSID,&LAGRBF_CellInfoSetUp));

  PetscFunctionReturn(PETSC_SUCCESS);
}