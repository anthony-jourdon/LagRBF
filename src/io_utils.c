#include <petsc.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

PetscErrorCode CreateDirectory(const char dirname[])
{
  PetscMPIInt rank;
  int num,error_number;

  PetscFunctionBeginUser;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Generate a new directory on proc 0 */
  error_number = 0;
  if (rank == 0) {
    num = mkdir(dirname, S_IRWXU | S_IRWXG);
    error_number = errno;
  }
  PetscCall(MPI_Bcast(&num,1,MPI_INT,0,PETSC_COMM_WORLD));
  PetscCall(MPI_Bcast(&error_number,1,MPI_INT,0,PETSC_COMM_WORLD));

  if (error_number == EEXIST) {
    PetscPrintf(PETSC_COMM_WORLD,"Writing output to existing directory: %s \n",dirname);
  } else if (error_number == EACCES) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Write permission is denied for the parent directory in which the new directory is to be added");
  } else if (error_number == EMLINK) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The parent directory has too many links (entries)");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The file system doesn't have enough room to create the new directory");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"The parent directory of the directory being created is on a read-only file system and cannot be modified");
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Created output directory: %s\n",dirname);
  }
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestDirectory(const char dirname[], char mode, PetscBool *_exists)
{
  PetscMPIInt rank;
  int i_exists = 0;
  PetscBool exists = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  if (rank == 0) {
    PetscCall(PetscTestDirectory(dirname,mode,&exists));
    i_exists = (int)exists;
  }
  PetscCallMPI(MPI_Bcast(&i_exists,1,MPI_INT,0,PETSC_COMM_WORLD));

  if (i_exists == 1) { *_exists = PETSC_TRUE; }
  else               { *_exists = PETSC_FALSE; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestFile(const char filename[], char mode, PetscBool *_exists)
{
  PetscMPIInt rank;
  int i_exists = 0;
  PetscBool exists = PETSC_FALSE;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  if (rank == 0) {
    PetscCall(PetscTestFile(filename,mode,&exists));
    i_exists = (int)exists;
  }
  PetscCallMPI(MPI_Bcast(&i_exists,1,MPI_INT,0,PETSC_COMM_WORLD));

  if (i_exists == 1) { *_exists = PETSC_TRUE; }
  else               { *_exists = PETSC_FALSE; }
  PetscFunctionReturn(0);
}

/* PVD helpers */
PetscErrorCode ParaviewPVDOpen(const char pvdfilename[])
{
  PetscMPIInt rank;
  FILE        *fp;
  PetscFunctionBeginUser;
  /* only master generates this file */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(rank != 0) { PetscFunctionReturn(0); }

  fp = fopen(pvdfilename,"w");
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open PVD file %s",pvdfilename);
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#ifdef WORDSIZE_BIGENDIAN
  fprintf(fp,"<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"BigEndian\">\n");
#else
  fprintf(fp,"<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
#endif

  fprintf(fp,"<Collection>\n");

  fprintf(fp,"</Collection>\n");
  fprintf(fp,"</VTKFile>");
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParaviewPVDAppend(
  const char pvdfilename[],
  double     time,
  const char datafile[], 
  const char directory_name[]
)
{
  PetscMPIInt rank;
  FILE        *fp;
  char        line[10000];
  int         key_L,position;
  char        key[] = "</Collection>";
  PetscFunctionBeginUser;
  /* only master generates this file */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank != 0) { PetscFunctionReturn(PETSC_SUCCESS); }

  fp = fopen(pvdfilename,"r+");
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open existing PVD file %s",pvdfilename);
  /* reset to start of file */
  rewind(fp);

  key_L = strlen(key);
  position = -1;
  while (!feof(fp)) {
    position = ftell(fp);
    PetscCheck(fgets(line,10000-1,fp),PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"fgets() failed");
    if (strncmp(key,line,key_L) == 0) {
      break;
    }
  }
  PetscCheck(position != -1,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot locate keyword in pvd file");

  fseek(fp,position,SEEK_SET);

  /* write new data */
  if (directory_name == NULL) {
    fprintf(fp,"  <DataSet timestep=\"%1.6e\" file=\"./%s\"/>\n",time, datafile );
  } else if ( (strlen(directory_name) == 0) ) {
    fprintf(fp,"  <DataSet timestep=\"%1.6e\" file=\"./%s\"/>\n",time, datafile );
  } else {
    fprintf(fp,"  <DataSet timestep=\"%1.6e\" file=\"%s/%s\"/>\n",time, directory_name, datafile );
  }

  /* close tag */
  fprintf(fp,"</Collection>\n");
  fprintf(fp,"</VTKFile>");

  fclose(fp);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParaviewPVDOpenAppend(
  PetscBool  not_first, 
  PetscInt   step, 
  const char pvdfilename[], 
  double     time, 
  const char datafile[], 
  const char directory_name[]
)
{
  PetscBool file_found;
  PetscFunctionBeginUser;
  if (!not_first) {
    if (step == 0) {
      PetscCall(ParaviewPVDOpen(pvdfilename));
      PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
    } else {
      TestFile(pvdfilename,'r',&file_found);
      if (file_found) {
        ParaviewPVDAppend(pvdfilename,time,datafile,directory_name);
      } else {
        ParaviewPVDOpen(pvdfilename);
        ParaviewPVDAppend(pvdfilename,time,datafile,directory_name);
      }
    }
  } else {
    TestFile(pvdfilename,'r',&file_found);
    if (!file_found) {
      ParaviewPVDOpen(pvdfilename);
      ParaviewPVDAppend(pvdfilename,time,datafile,directory_name);
    } else {
      ParaviewPVDAppend(pvdfilename,time,datafile,directory_name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLoadFromFile(Vec v, const char *name)
{
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)v),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer,name));
  PetscCall(VecLoad(v,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscOptionsClearValue(NULL,"-vecload_block_size"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ISLoadFromFile(IS is, const char *name)
{
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)is),&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer,name));
  PetscCall(ISLoad(is,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#if 0
PetscErrorCode GetOutputFieldsFromOptions(PetscInt *nfields, char ***fields)
{
  PetscBool found;
  char      prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"output_"));
  PetscCall(PetscOptionsGetStringArray(NULL,prefix,"-output_fields",NULL,nfields,&found));
  if (found) {
    PetscCall(PetscMalloc1(*nfields,fields));
    PetscCall(PetscOptionsGetStringArray(NULL,prefix,"-output_fields",*fields,nfields,&found));
  } else {
    *nfields = 0;
    *fields  = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif