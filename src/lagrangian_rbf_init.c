#include <petsc.h>

PetscClassId LAGRBF_CLASSID;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;
extern PetscLogEvent LAGRBF_PartitionAssembleRBFMatrix;
extern PetscLogEvent LAGRBF_PartitionFactorRBFMatrix;
extern PetscLogEvent LAGRBF_PartitionMatSetRHS;
extern PetscLogEvent LAGRBF_PartitionSolveLocalCoefficients;
extern PetscLogEvent LAGRBF_PartitionSolveInterpolationCoefficients;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK1;
extern PetscLogEvent LAGRBF_AdvectionExplicit_RK4;

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

  /* Interpolation */
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_Scalar",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_Scalar));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_AVX256",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_AVX256));
  /* Coefficients solve */
  PetscCall(PetscLogEventRegister("PartitionAssembleRBFMatrix",LAGRBF_CLASSID,&LAGRBF_PartitionAssembleRBFMatrix));
  PetscCall(PetscLogEventRegister("PartitionFactorRBFMatrix",LAGRBF_CLASSID,&LAGRBF_PartitionFactorRBFMatrix));
  PetscCall(PetscLogEventRegister("PartitionMatSetRHS",LAGRBF_CLASSID,&LAGRBF_PartitionMatSetRHS));
  PetscCall(PetscLogEventRegister("PartitionSolveLocalCoefficients",LAGRBF_CLASSID,&LAGRBF_PartitionSolveLocalCoefficients));
  PetscCall(PetscLogEventRegister("PartitionSolveInterpolationCoefficients",LAGRBF_CLASSID,&LAGRBF_PartitionSolveInterpolationCoefficients));
  /* Advection */
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK1",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK1));
  PetscCall(PetscLogEventRegister("AdvectionExplicit_RK4",LAGRBF_CLASSID,&LAGRBF_AdvectionExplicit_RK4));

  PetscFunctionReturn(PETSC_SUCCESS);
}