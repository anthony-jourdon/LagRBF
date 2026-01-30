#include "lagrangian_rbf.h"

PetscErrorCode StagGetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps)
{
  PetscInt  nt,nn;
  PetscInt  *tsteps_int = NULL;
  PetscReal *tsteps = NULL;
  PetscBool found;
  char      prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"stag_"));
  PetscCall(PetscOptionsGetInt(NULL,prefix,"-nsteps",&nt,&found));
  if (!found) {
    if (nsteps) *nsteps = 0;
    PetscCalloc1(1,&tsteps);
    if (time_steps) *time_steps = tsteps;
    PetscCalloc1(1,&tsteps_int);
    if (steps) *steps = tsteps_int;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1(nt,&tsteps));
  PetscCall(PetscMalloc1(nt,&tsteps_int));
  nn = nt;
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-time_steps",tsteps,&nn,&found));
  PetscCheck(nn == nt,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Number of time steps does not match the length of the time_steps array");
  nn = nt;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-steps",tsteps_int,&nn,&found));
  PetscCheck(nn == nt,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Number of time steps does not match the length of the steps array");
  if (nsteps)     *nsteps = nt;
  if (time_steps) *time_steps = tsteps;
  if (steps)      *steps = tsteps_int;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagRestoreTimeSteps(PetscInt **steps, PetscReal **time_steps)
{
  PetscReal *tsteps = *time_steps;
  PetscInt  *tsteps_int = *steps;
  PetscFunctionBeginUser;
  PetscCall(PetscFree(tsteps));
  PetscCall(PetscFree(tsteps_int));
  *time_steps = NULL;
  *steps = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagGetFilelistFromOptions(PetscInt step, PetscInt *nfiles, char ***filenames)
{
  PetscInt    nt,nn,f;
  char        **fnames = NULL, **fieldnames = NULL;
  char        prefix[PETSC_MAX_PATH_LEN],directory[PETSC_MAX_PATH_LEN],model[PETSC_MAX_PATH_LEN];
  PetscBool   found;
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"stag_"));
  PetscCall(PetscOptionsGetString(NULL,prefix,"-directory",directory,PETSC_MAX_PATH_LEN-1,&found));
  PetscCall(PetscOptionsGetString(NULL,prefix,"-model",model,PETSC_MAX_PATH_LEN-1,&found));

  PetscCall(PetscOptionsGetInt(NULL,prefix,"-nfiles",&nt,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snfiles not found. User must provide the number of files to read.",prefix);
  PetscCall(PetscMalloc1(nt,&fieldnames));
  nn = nt;
  PetscCall(PetscOptionsGetStringArray(NULL,prefix,"-field_names", fieldnames, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%sfield_names not found. User must provide the list of field names to read.",prefix);
  PetscCheck(nn == nt,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sfield_names must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,nt,nn);
  
  PetscCall(PetscMalloc1(nt,&fnames));
  for (f=0; f<nt; f++) {
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fnames[f]));
    PetscCall(PetscSNPrintf(fnames[f],PETSC_MAX_PATH_LEN-1,"%s/%s_%s%05" PetscInt_FMT,directory,model,fieldnames[f],step));
  }
  if (nfiles)    *nfiles = nt;
  if (filenames) *filenames = fnames;
  PetscCall(PetscFree(fieldnames));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagRestoreFilelist(PetscInt nfiles, char ***filenames)
{
  PetscInt f;
  char     **fnames = *filenames;
  PetscFunctionBeginUser;
  for (f=0; f<nfiles; f++) {
    PetscCall(PetscFree(fnames[f]));
  }
  PetscCall(PetscFree(fnames));
  *filenames = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}