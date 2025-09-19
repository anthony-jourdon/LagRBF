#include <petsc.h>
#include "partition_of_unity.h"
#include "partition.h"

PetscErrorCode PUCreate(PartitionOfUnity *pu)
{
  PartitionOfUnity p;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&p));
  PetscCall(PetscSNPrintf(p->name, PETSC_MAX_PATH_LEN-1,"pu_mesh"));
  p->dap = NULL;
  p->dim = 0;
  p->n = NULL;
  p->npart = 0;
  p->O = NULL;
  p->L = NULL;

  *pu = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUSetFromOptions(PartitionOfUnity pu)
{
  PetscInt  nn,d;
  char      prefix[PETSC_MAX_PATH_LEN],option_name[PETSC_MAX_PATH_LEN];
  PetscBool found;
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PUCreate() first.");
  /* Set prefix for options */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN-1, "%s_", pu->name));

  /* Spatial dimension */
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sdim", prefix));
  PetscCall(PetscOptionsGetInt(NULL,NULL,option_name,&pu->dim,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not provided. Use it to set the spatial dimension of the problem.",option_name);
  PetscCheck(pu->dim >= 1 && pu->dim <= 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Spatial dimension must be between 1 and 3. Found: %" PetscInt_FMT,pu->dim);
  
  /* Number of partition in each dimension */
  PetscCall(PetscMalloc1(pu->dim,&pu->n));
  nn = pu->dim; // Number of entries expected
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sn", prefix));
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,pu->n,&nn,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option %s not provided. Use it to set the number of partitions in each spatial dimension.",option_name);
  PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn);
  
  /* Bounding box */
  /* Origin */
  PetscCall(PetscMalloc1(pu->dim,&pu->O));
  for (d=0; d<pu->dim; d++) { pu->O[d] = 0.0; } // Default origin
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sO",prefix));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,option_name,pu->O,&nn,&found));
  if (found) { PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn); }
  /* Length */
  PetscCall(PetscMalloc1(pu->dim,&pu->L));
  for (d=0; d<pu->dim; d++) { pu->L[d] = 1.0; } // Default length
  PetscCall(PetscSNPrintf(option_name, PETSC_MAX_PATH_LEN-1, "-%sL",prefix));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,option_name,pu->L,&nn,&found));
  if (found) { PetscCheck(nn == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option %s. Expected %" PetscInt_FMT " entries, found %" PetscInt_FMT "",option_name,pu->dim,nn); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PULoadFromFile(PartitionOfUnity pu, const char filename[])
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUCreateDMDA(PartitionOfUnity pu)
{
  DM       da;
  PetscInt stencil_width=1;
  char     prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  switch (pu->dim) {
    case 1:
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,pu->n[0],1,stencil_width,NULL,&da));
    break;

    case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,pu->n[0],pu->n[1],PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,&da));
    break;

    case 3: 
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,pu->n[0],pu->n[1],pu->n[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,stencil_width,NULL,NULL,NULL,&da));
    break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Dimension can only be 1, 2 or 3. Found: %" PetscInt_FMT,pu->dim);
  }
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(PetscObjectSetName((PetscObject)da,pu->name));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN-1, "%s_",pu->name));
  PetscCall(DMSetOptionsPrefix(da,prefix));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,pu->O[0],pu->L[0],pu->O[1],pu->L[1],pu->O[2],pu->L[2]));
  
  pu->dap = da;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionOfUnityCreate(PartitionOfUnity *pu)
{
  PartitionOfUnity p;
  DM               da=NULL;
  PetscInt         d,s[3],n[3];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUCreate(&p));
  PetscCall(PUSetFromOptions(p));
  PetscCall(PUCreateDMDA(p));
  /* set the total number of partitions (local) */
  PetscCall(PUGetDMDA(p,&da));
  PetscCall(DMDAGetCorners(da,&s[0],&s[1],&s[2],&n[0],&n[1],&n[2]));
  p->npart = 1; // Initialize
  for (d=0; d<p->dim; d++) { p->npart *= n[d]; }

  *pu = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PartitionOfUnityDestroy(PartitionOfUnity *pu)
{
  PartitionOfUnity p = *pu;
  PetscFunctionBeginUser;

  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (p->dap) { PetscCall(DMDestroy(&p->dap)); }
  if (p->n)   { PetscCall(PetscFree(p->n)); }
  if (p->O)   { PetscCall(PetscFree(p->O)); }
  if (p->L)   { PetscCall(PetscFree(p->L)); }
  PetscCall(PetscFree(p));
  *pu = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetDMDA(PartitionOfUnity pu, DM *dap)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  PetscCheck(pu->dap,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Partition of Unity mesh not created yet. Call PUCreateDMDA() first.");
  if (dap) { *dap = pu->dap; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetNPartitions(PartitionOfUnity pu, PetscInt *npart)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  if (npart) { *npart = pu->npart; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PUGetDimension(PartitionOfUnity pu, PetscInt *dim)
{
  PetscFunctionBeginUser;
  PetscCheck(pu,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Partition of Unity structure is NULL. Call PartitionOfUnityCreate() first.");
  if (dim) { *dim = pu->dim; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  View the Mesh associated with the Partition Of Unity DMDA.
  Mostly for debugging purposes.
  Use options to control the output.
  Example usage for a vts output: 
    -pu_mesh_view vtk:pu_mesh.vts:VTK_VTS:write
  See PetscOptionsCreateViewer() manual page for more details.
*/
PetscErrorCode PartitionOfUnityMeshView(PartitionOfUnity pu)
{
  DM                dap=NULL;
  Vec               testvec;
  PetscViewer       viewer;
  PetscViewerFormat format;
  const char        *prefix;
  PetscFunctionBeginUser;

  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dap,&prefix));
  PetscCall(PetscOptionsCreateViewer(PETSC_COMM_WORLD,NULL,prefix,"-view",&viewer,&format,NULL));
  /* Create a vec just to print out the mesh */
  PetscCall(DMCreateGlobalVector(dap,&testvec));
  PetscCall(VecSet(testvec,1.0));
  PetscCall(VecView(testvec,viewer));
  PetscCall(VecDestroy(&testvec));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}