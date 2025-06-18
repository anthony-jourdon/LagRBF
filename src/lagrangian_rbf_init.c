#include <petsc.h>

PetscClassId LAGRBF_CLASSID;
extern PetscLogEvent LAGRBF_PointwiseInterpolation_Scalar;

PetscErrorCode LagRBFInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(argc,args,file,help));
  PetscCall(PetscLogDefaultBegin());

  PetscCall(PetscLogEventRegister("PointwiseInterpolation_Scalar",LAGRBF_CLASSID,&LAGRBF_PointwiseInterpolation_Scalar));


  PetscFunctionReturn(PETSC_SUCCESS);
}