#include <petsc.h>

PetscClassId LAGRBF_CLASSID;
extern PetscLogEvent LAGRBF_SwarmInterpSetWeights_Scalar;
extern PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX256;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;
extern PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix;
extern PetscLogEvent LAGRBF_FactorRBFMatrix;
extern PetscLogEvent LAGRBF_MatSetRHS_Fac;
extern PetscLogEvent LAGRBF_MatSetRHS_KSP;
extern PetscLogEvent LAGRBF_SolveLocalCoefficients_Fac;
extern PetscLogEvent LAGRBF_SolveLocalCoefficients_KSP;
extern PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK1;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK2;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK4;
extern PetscLogEvent LAGRBF_AdvectionExplicit_DOPRI;
extern PetscLogEvent LAGRBF_SphericalCoordinates_Halton;

static PetscErrorCode LagRBFWriteInitialInfo(void)
{
  PetscFunctionBeginUser;

#if defined(__AVX__)
  PetscPrintf(PETSC_COMM_WORLD,"** AVX detected - optimized kernels will be used\n");
#else
  PetscPrintf(PETSC_COMM_WORLD,"** AVX not detected - optimized kernels will not be used.\n");
  PetscPrintf(PETSC_COMM_WORLD,"** If your system supports AVX, consider adding options,\n");
  PetscPrintf(PETSC_COMM_WORLD,"** e.g. -march=native, to LAG_CFLAGS in makefile.arch\n");
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LagRBFInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(argc,args,file,help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(LagRBFWriteInitialInfo());

  /* Register the class */
  PetscCall(PetscClassIdRegister("Lagrangian RBF", &LAGRBF_CLASSID));

  /* Interpolation */
  PetscCall(PetscLogEventRegister("SwarmInterpSetWeights_Scalar",LAGRBF_CLASSID,&LAGRBF_SwarmInterpSetWeights_Scalar));
  PetscCall(PetscLogEventRegister("SwarmInterpSetWeights_AVX256",LAGRBF_CLASSID,&LAGRBF_SwarmInterpSetWeights_AVX256));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_Scalar",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_Scalar));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_AVX256",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_AVX256));
  /* Coefficients solve */
  PetscCall(PetscLogEventRegister("PartitionAssembleRBFMatrix",LAGRBF_CLASSID,&LAGRBF_PartitionAssembleRBFMatrix));
  PetscCall(PetscLogEventRegister("FactorRBFMatrix",LAGRBF_CLASSID,&LAGRBF_FactorRBFMatrix));
  PetscCall(PetscLogEventRegister("MatSetRHS_Fac",LAGRBF_CLASSID,&LAGRBF_MatSetRHS_Fac));
  PetscCall(PetscLogEventRegister("MatSetRHS_KSP",LAGRBF_CLASSID,&LAGRBF_MatSetRHS_KSP));
  PetscCall(PetscLogEventRegister("SolveLocalCoefficients_Fac",LAGRBF_CLASSID,&LAGRBF_SolveLocalCoefficients_Fac));
  PetscCall(PetscLogEventRegister("SolveLocalCoefficients_KSP",LAGRBF_CLASSID,&LAGRBF_SolveLocalCoefficients_KSP));
  PetscCall(PetscLogEventRegister("PartitionSolveInterpolationCoefficients",LAGRBF_CLASSID,&LAGRBF_PartitionSolveInterpolationCoefficients));
  /* Advection */
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK1",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK1));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK2",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK2));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK4",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK4));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_DOPRI",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_DOPRI));

  /* Utilities */
  PetscCall(PetscLogEventRegister("SphericalCoordinates_Halton",LAGRBF_CLASSID,&LAGRBF_SphericalCoordinates_Halton));

  PetscFunctionReturn(PETSC_SUCCESS);
}