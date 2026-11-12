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
  file:    io_utils.c

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
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

PetscErrorCode CreateDirectory(const char dirname[])
{
  PetscMPIInt rank;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Create the output directory */
  if (rank == 0) {
    PetscBool exists;
    PetscCall(PetscTestDirectory(dirname,'w',&exists));
    if (!exists) {
      PetscCall(PetscMkdir(dirname));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Created output directory %s\n",dirname));
    }
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PVD helpers */
PetscErrorCode ParaviewPVDOpen(const char pvdfilename[])
{
  PetscMPIInt rank;
  FILE        *fp;
  PetscFunctionBeginUser;
  /* only master generates this file */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(rank != 0) { PetscFunctionReturn(PETSC_SUCCESS); }

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
  PetscMPIInt rank;
  PetscBool   file_found;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank != 0) { PetscFunctionReturn(PETSC_SUCCESS); }
  if (!not_first) {
    if (step == 0) {
      PetscCall(ParaviewPVDOpen(pvdfilename));
      PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
    } else {
      //PetscCall(TestFile(pvdfilename,'r',&file_found));
      PetscCall(PetscTestFile(pvdfilename,'r',&file_found));
      if (file_found) {
        PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
      } else {
        PetscCall(ParaviewPVDOpen(pvdfilename));
        PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
      }
    }
  } else {
    //PetscCall(TestFile(pvdfilename,'r',&file_found));
    PetscCall(PetscTestFile(pvdfilename,'r',&file_found));
    if (!file_found) {
      PetscCall(ParaviewPVDOpen(pvdfilename));
      PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
    } else {
      PetscCall(ParaviewPVDAppend(pvdfilename,time,datafile,directory_name));
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

PetscErrorCode GetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps)
{
  PetscInt  nt,nn;
  PetscInt  *tsteps_int = NULL;
  PetscReal *tsteps = NULL;
  PetscBool found;
  char      prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"data_"));
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

PetscErrorCode RestoreTimeSteps(PetscInt **steps, PetscReal **time_steps)
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

static PetscErrorCode StrFillFromToken(
  const char *token, 
  size_t token_len,
  const char *eos,
  char out[], 
  PetscInt *out_pos, 
  const char **template_pos
)
{
  PetscInt   i;
  const char *str = token ? token : "";
  size_t     slen;
  PetscFunctionBeginUser;

  /* Copy the content of the token */
  PetscCall(PetscStrlen(str,&slen));
  for (i=0; i<slen; i++) {
    PetscCheck(*out_pos < PETSC_MAX_PATH_LEN-1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expanded filename exceeds PETSC_MAX_PATH_LEN");
    out[*out_pos] = str[i];
    (*out_pos)++;
  }
  /* Append the end-of-string marker if any */
  if (eos && eos[0] != '\0') {
    if (slen > 0 && str[slen-1] != eos[0]) {
      PetscCheck(*out_pos < PETSC_MAX_PATH_LEN-1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expanded filename exceeds PETSC_MAX_PATH_LEN");
      out[(*out_pos)++] = eos[0];
    }
  }
  /* Append the null terminator */
  out[*out_pos] = '\0';
  /* Update the template position */
  *template_pos += token_len;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildFilenameFromTemplate(
  const char template_name[],
  const char directory[],
  const char prefix[],
  const char field[],
  const char separator[],
  PetscInt   step,
  char       out[]
)
{
  /* 
  Example: "{directory}{prefix}{step:06}{separator}{field}" -> "data/model000042_temp" 
  Example: "{directory}{prefix}{separator}{field}{step:05}" -> "data/model_temp00042"
  */
  PetscInt   out_pos = 0;
  PetscBool  match;
  char       stepbuf[64];
  const char *p;

  PetscFunctionBeginUser;
  PetscCheck(template_name,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"template_name must not be NULL");
  PetscCheck(out,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"out must not be NULL");

  out[0] = '\0';
  p = template_name;

  while (*p) {
    if (*p != '{') {
      PetscCheck(out_pos < PETSC_MAX_PATH_LEN-1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expanded filename exceeds PETSC_MAX_PATH_LEN");
      out[out_pos++] = *p++;
      out[out_pos] = '\0';
      continue;
    }

    PetscCall(PetscStrncmp(p,"{directory}",11,&match));
    if (match) {
      PetscCall(StrFillFromToken(directory, 11, "/", out, &out_pos, &p));
      continue;
    }
    PetscCall(PetscStrncmp(p,"{prefix}",8,&match));
    if (match) {
      PetscCall(StrFillFromToken(prefix, 8, NULL, out, &out_pos, &p));
      continue;
    }
    PetscCall(PetscStrncmp(p,"{field}",7,&match));
    if (match) {
      PetscCall(StrFillFromToken(field, 7, NULL, out, &out_pos, &p));
      continue;
    }
    PetscCall(PetscStrncmp(p,"{separator}",11,&match));
    if (match) {
      PetscCall(StrFillFromToken(separator, 11, NULL, out, &out_pos, &p));
      continue;
    }
    PetscCall(PetscStrncmp(p,"{step}",6,&match));
    if (match) {
      PetscCall(PetscSNPrintf(stepbuf,sizeof(stepbuf),"%" PetscInt_FMT,step));
      PetscCall(StrFillFromToken(stepbuf, 6, NULL, out, &out_pos, &p));
      continue;
    }
    PetscCall(PetscStrncmp(p,"{step:",6,&match));
    if (match) {
      const char *q = p + 6;
      int        width = 0;
      PetscBool  has_digit = PETSC_FALSE;
      size_t     slen;
      PetscInt   i;

      while (*q >= '0' && *q <= '9') {
        has_digit = PETSC_TRUE;
        width = 10 * width + (int)(*q - '0');
        q++;
      }
      PetscCheck(has_digit && *q == '}',PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid step token. Expected {step:N} where N is an integer width");
      PetscCheck(width >= 0 && width <= 50,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Step padding width must be in [0,50]");

      PetscCall(PetscSNPrintf(stepbuf,sizeof(stepbuf),"%0*" PetscInt_FMT,width,step));
      PetscCall(PetscStrlen(stepbuf,&slen));
      for (i=0; i<slen; i++) {
        PetscCheck(out_pos < PETSC_MAX_PATH_LEN-1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expanded filename exceeds PETSC_MAX_PATH_LEN");
        out[out_pos++] = stepbuf[i];
      }
      out[out_pos] = '\0';
      p = q + 1;
      continue;
    }

    PetscCheck(PETSC_FALSE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown template token near: %.32s",p);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetFilelistFromOptions(const char *prefix, PetscInt step, PetscInt *nfiles, char ***filenames, char **coorfname)
{
  PetscInt    nt,nn,f;
  char        *coor_fname = NULL;
  char        **fnames = NULL, **fieldnames = NULL;
  char        template[PETSC_MAX_PATH_LEN],template_dir[PETSC_MAX_PATH_LEN];
  char        name_prefix[PETSC_MAX_PATH_LEN],separator[PETSC_MAX_PATH_LEN];
  char        directory[PETSC_MAX_PATH_LEN],dir_prefix[PETSC_MAX_PATH_LEN],basename[PETSC_MAX_PATH_LEN],sub_dir[PETSC_MAX_PATH_LEN];
  char        coor_name[PETSC_MAX_PATH_LEN];
  PetscBool   found;
  PetscFunctionBeginUser;

  /* Default template -> StagYY naming convention: outdir/model_t00039 */
  PetscCall(PetscSNPrintf(template, PETSC_MAX_PATH_LEN-1, "{directory}{prefix}{separator}{field}{step:05}"));
  PetscCall(PetscOptionsGetString(NULL,prefix,"-fname_template",template,PETSC_MAX_PATH_LEN-1,&found));
  if (!found) { PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Using default filename template: %s\n",template)); }

  /* Initialize all strings to empty characters */
  name_prefix[0] = '\0';
  separator[0]   = '\0';
  PetscCall(PetscOptionsGetString(NULL,prefix,"-fname_prefix",name_prefix,PETSC_MAX_PATH_LEN-1,&found));
  PetscCall(PetscOptionsGetString(NULL,prefix,"-fname_separator",separator,PETSC_MAX_PATH_LEN-1,&found));

  /* Subdirectory template when a step based directory structure is used. Default is empty */
  template_dir[0] = '\0';
  PetscCall(PetscOptionsGetString(NULL,prefix,"-dirname_template",template_dir,PETSC_MAX_PATH_LEN-1,&found));
  if (!found) { PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Using default directory template: %s\n",template_dir)); }
  dir_prefix[0] = '\0';
  PetscCall(PetscOptionsGetString(NULL,prefix,"-dirname_prefix",dir_prefix,PETSC_MAX_PATH_LEN-1,&found));
  PetscCall(BuildFilenameFromTemplate(template_dir, "", dir_prefix, "", "", step, sub_dir));
  /* Absolute directory path */
  basename[0] = '\0';
  PetscCall(PetscOptionsGetString(NULL,prefix,"-dirname_base",basename,PETSC_MAX_PATH_LEN-1,&found));
  if (found) {
    PetscCall(PetscSNPrintf(directory, PETSC_MAX_PATH_LEN-1, "%s/%s", basename, sub_dir));
  } else {
    PetscCall(PetscSNPrintf(directory, PETSC_MAX_PATH_LEN-1, "%s", sub_dir));
  }

  /* Fields */
  PetscCall(PetscOptionsGetInt(NULL,prefix,"-fname_nfiles",&nt,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%sfname_nfiles not found. User must provide the number of files to read.",prefix);
  PetscCall(PetscMalloc1(nt,&fieldnames));
  nn = nt;
  PetscCall(PetscOptionsGetStringArray(NULL,prefix,"-fname_fields", fieldnames, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%sfname_fields not found. User must provide the list of field names to read.",prefix);
  PetscCheck(nn == nt,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sfname_fields must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,nt,nn);
  PetscCall(PetscMalloc1(nt,&fnames));

  /* Assemble filenames using template */
  for (f=0; f<nt; f++) {
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fnames[f]));
    PetscCall(BuildFilenameFromTemplate(template, directory, name_prefix, fieldnames[f], separator, step, fnames[f]));
  }

  /* Coordinate filename */
  coor_name[0] = '\0';
  PetscCall(PetscOptionsGetString(NULL,prefix,"-fname_coordinates",coor_name,PETSC_MAX_PATH_LEN-1,&found));
  if (found) { 
    PetscMalloc1(PETSC_MAX_PATH_LEN,&coor_fname);
    PetscCall(BuildFilenameFromTemplate(template, directory, name_prefix, coor_name, separator, step, coor_fname)); 
  }

  if (nfiles)    *nfiles = nt;
  if (filenames) *filenames = fnames;
  if (coorfname) *coorfname = coor_fname;
  PetscCall(PetscFree(fieldnames));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreFilelist(PetscInt nfiles, char ***filenames, char **coorfname)
{
  PetscInt f;
  char     **fnames = *filenames;
  PetscFunctionBeginUser;
  if (fnames) {
    for (f=0; f<nfiles; f++) {
    PetscCall(PetscFree(fnames[f]));
    }
    PetscCall(PetscFree(fnames));
    *filenames = NULL;
  }
  if (coorfname) {
    PetscCall(PetscFree(*coorfname));
    *coorfname = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}