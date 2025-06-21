#include <petsc.h>

PetscClassId LAGRBF_CLASSID;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_AVX256;

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

  PetscCall(PetscLogEventRegister("PointwiseInterpolation_Scalar",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_Scalar));
  PetscCall(PetscLogEventRegister("PointwiseInterpolation_AVX256",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_AVX256));

  PetscFunctionReturn(PETSC_SUCCESS);
}