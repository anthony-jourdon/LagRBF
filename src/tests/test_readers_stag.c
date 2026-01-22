static const char help[] = "Test to read HDF5 files.\n\n";

#include <petsc.h>
#include "readers/reader_utils.h"
#include "readers/stagYY/hdf5_read_coor.h"
#include "readers/stagYY/hdf5_read_fields.h"
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "io_utils.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_data.h"
#include "swarm/swarm_interp.h"
#include "swarm/swarm_output.h"
#include "partitions/partition.h"
#include "partitions/partition_solve.h"
#include "partition_of_unity.h"
#include "interpolation.h"
#include "cell_info.h"
#include "advection/advection_ctx.h"

#if 0
static PetscErrorCode StagReshapeVelocity(PetscInt dim, StagFieldInfo fdata, PetscReal *velocity)
{
  PetscInt p,d;
  PetscFunctionBeginUser;
  for (p=0; p<fdata->npoints; p++) {
    for (d=0; d<dim; d++) {
      velocity[ dim*p + d ] = fdata->field[ fdata->bs*p + d ];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  In parallel I don't know if can I do this better:
    Data (coordinates + fields) are loaded from HDF5 files.
    There is 1 file per StagYY mpi rank.
    I distribute these files among MY ranks (not stagYY) trying to balance the load of files between ranks.
    Nothing guaranties (and in general it is not guaranteed) that the points will end on the rank they have been loaded
    once SwarmMigrate() is called.
    Problem is that because I need to load fields every time steps, some points may not be on the rank they were originally loaded on.
    The only work around I found so far, is to re-size the swarm, reload the coords, load the fields and then call SwarmMigrate().
    But it means that this re-size is done at each time step. I don't know yet if it's a problem or not.
*/
static PetscErrorCode SwarmDataSetFields(DM swarm, StagFieldInfo cinfo, StagFieldInfo fdata)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;
  PetscReal     *coords,*velocity;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(swarm,&dim));
  /* Set the rank-local number of points */
  PetscCall(DMSwarmSetLocalSizes(swarm, cinfo->npoints, -1));
  /* Set the coordinate field */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  /* Access fields */
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, "fields", NULL, NULL, (void **)&velocity));
  /* Set coordinates */
  PetscCall(StagReshapeCoordinates(dim, cinfo, coords));
  /* Set velocity */
  PetscCall(StagReshapeVelocity(dim,fdata,velocity));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&velocity));
  /* Update cell id and rank */
  PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  StagFieldInfo    field_data,cinfo;
  StagGridType     coor_type;
  DM               swarm_data;
  PetscInt         d,f,nfiles,nn,step,step_start,step_end,nsteps;
  PetscReal        time,dt;
  PetscReal        *time_entries;
  //PetscInt         npoints,ndims,nsteps,tstart;
  //PetscReal        box_length,overlap,radius,epsilon;
  PetscBool        found,exists;
  char             **fname_list,**coor_dset;
  char             **vfname_list; // velocity file names
  const char       *fields[] = {"fields"};
  PetscFunctionBeginUser;
  
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nfiles",&nfiles,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -nfiles not found. User must provide the number of files to read.");
  PetscCall(PetscMalloc1(nfiles, &fname_list)); 
  PetscCall(PetscMalloc1(nfiles, &vfname_list)); 
  PetscCall(PetscMalloc1(3, &coor_dset));
  for (f=0; f<nfiles; f++) { 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&fname_list[f])); 
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&vfname_list[f]));
  }
  for (d=0; d<3; d++) {
    PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN,&coor_dset[d]));
  }
  /* Coordinates */
  nn = nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_fname", fname_list, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -coor_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -coor_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",nfiles,nn);
  nn = 3;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-coor_names", coor_dset, &nn, &found));
  /* Coordinate type, if nothing is given, the default is STAG_3D i.e., x,y,z normally ordered */
  coor_type = STAG_3D;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-coor_type",(PetscInt*)&coor_type,&found));

  /* Velocities */
  nn = nfiles;
  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-velo_fname", vfname_list, &nn, &found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -velo_fname not found. User must provide the list of files to read.");
  PetscCheck(nn == nfiles,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -velo_fname must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",nfiles,nn);
  
  /* Steps */
  step_start = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_start",&step_start,&found));
  step_end = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step_end",&step_end,&found));
  nsteps = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time_nsteps",&nsteps,&found));
  PetscMalloc1(nsteps,&time_entries);
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-time_entries",time_entries,&nsteps,&found));

  #if 0
  /* Backward loop time */
  for (step=step_end-1; step>=step_start; step--) {
    if (step-1 < step_start) { dt = 0.0; }
    else                     { dt = time_entries[step-1] - time_entries[step]; }
    PetscPrintf(PETSC_COMM_WORLD,"[Step %d] Time: %g, dt = %g\n",step,time_entries[step],dt);
  }

  PetscPrintf(PETSC_COMM_WORLD,"\n");
  /* Forward loop time */
  for (step=step_start; step<step_end; step++) {
    if (step == step_end-1) { dt = 0.0; }
    else                    { dt = time_entries[step+1] - time_entries[step]; }
    PetscPrintf(PETSC_COMM_WORLD,"[Step %d] Time: %g, dt = %g\n",step,time_entries[step],dt);
  }
#endif
#if 1

  /*
  Extract information from files and store into data structure
  Each rank reads a subset of files distributed among all ranks
  This is done only once for coordinates but other fields are only initialized, the data are read in the time loop
  */
  StagProcessCoordinates(PETSC_COMM_WORLD,(const char**)fname_list,nfiles,coor_dset,coor_type,&cinfo);
  HDF5_SetUpFieldData(PETSC_COMM_WORLD,(const char**)vfname_list,nfiles,&field_data);

  /* Create Partition mesh and data structures */
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }
  ctx->mtype = GEODYNAMICS; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Create data points */
  PetscCall(SwarmDataCreate(ctx,&swarm_data));

  for (step=step_start; step<step_end; step++) {
    time += step;
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(LagRBFSetTime(ctx,time));

    HDF5_GetFieldData(PETSC_COMM_WORLD, (const char **)vfname_list, nfiles, step, field_data);

    PetscCall(SwarmDataSetFields(swarm_data, cinfo, field_data));
    /* Output data */
    PetscCall(SwarmOutput(ctx,swarm_data,1,fields,PETSC_FALSE));

  }

  


  for (f=0; f<nfiles; f++) { 
    PetscCall(PetscFree(fname_list[f])); 
    PetscCall(PetscFree(vfname_list[f]));
  }
  for (d=0; d<3; d++) { PetscCall(PetscFree(coor_dset[d])); }
  PetscCall(PetscFree(fname_list));
  PetscCall(PetscFree(vfname_list));
  PetscCall(PetscFree(coor_dset));
  PetscCall(PetscFree(time_entries));
  StagFieldInfoDestroy(&field_data);
  StagFieldInfoDestroy(&cinfo);
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

typedef struct {
  int    n[3];
  int    nblocks;
  int    bs;
  double *data;
  float  *x,*y,*z;
  float  rcmb;
} StagField;

static PetscErrorCode TestBinaryStagg(StagField *sfield)
{
  FILE       *fp;
  PetscInt   nval;
  int        ib,ix,iy,iz,i,j,k,b,v,magic,xyp,nz2,npi,step;
  int        ncpu[4],ntot[4],npar[4];
  float      rcmb,time,erupta_total,botT_val,scale_fac=1.0;
  float      aspect[2];
  float      *zg,*x,*y,*z,*data_local;
  double     *data;
  const char fname[] = "/mnt/c/Users/jourdon/Documents/Scripts/PostProc/StagYY/PJB6_YS1_Rh32_vp00039";
  //const char fname[] = "/mnt/c/Users/jourdon/Documents/Scripts/PostProc/StagYY/PJB6_YS1_Rh32_t00039";
  PetscFunctionBeginUser;

  /* Data block-size here velocity and pressure */
  nval = 4;
  //nval = 1;
  fp = fopen(fname,"rb");
  PetscCheck(fp != NULL,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot open file: %s",fname);
  
  fread(&magic,sizeof(int),1,fp);
  PetscPrintf(PETSC_COMM_WORLD,"Magic number: %d\n",magic);
  if ( (magic < 100 && nval > 1) || (magic > 300 && nval == 1) ) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"File %s contains the wrong number of components in provided field.",fname);
  }
  magic = magic % 100;
  PetscPrintf(PETSC_COMM_WORLD,"Magic modulo 100: %d\n",magic);
  if (magic >= 9 && nval == 4) { xyp = 1; } // extra ghost point in x & y directions
  else                         { xyp = 0; }

  fread(ntot,sizeof(int),4,fp); /* Total number of grid points + # blocks (2 for yin-yang grid) */
  fread(aspect,sizeof(float),2,fp); /* Aspect ratios */
  fread(npar,sizeof(int),4,fp); /* Number of parallel subdomains */
  // z-coordinates of what ??
  nz2 = 2*ntot[2] + 1;
  zg = (float*) malloc( nz2 * sizeof(float) );
  fread(zg,sizeof(float),nz2,fp);
  /* Compute nx, ny, nz and nb PER CPU */
  for (k=0; k<4; k++) { ncpu[k] = ntot[k] / npar[k]; }
  /* Number of values per "read" block */
  npi = (ncpu[0]+xyp) * (ncpu[1]+xyp) * ncpu[2] * ncpu[3] * nval;
  fread(&rcmb,sizeof(float),1,fp); /* distance from surface to CMB */
  fread(&step,sizeof(int),1,fp); /* time step */
  fread(&time,sizeof(float),1,fp); /* simulation time */
  fread(&erupta_total,sizeof(float),1,fp); /* ?? */
  fread(&botT_val,sizeof(float),1,fp); /* bottom temperature value */

  // Report for debugging
  PetscPrintf(PETSC_COMM_WORLD,"ntot: %d %d %d %d\n",ntot[0],ntot[1],ntot[2],ntot[3]);
  PetscPrintf(PETSC_COMM_WORLD,"Aspect ratios: %g %g\n",aspect[0],aspect[1]);
  PetscPrintf(PETSC_COMM_WORLD,"npar: %d %d %d %d\n",npar[0],npar[1],npar[2],npar[3]);
  PetscPrintf(PETSC_COMM_WORLD,"n per CPU: %d %d %d %d\n",ncpu[0],ncpu[1],ncpu[2],ncpu[3]);
  PetscPrintf(PETSC_COMM_WORLD,"Number of values per block: %d\n",npi);
  PetscPrintf(PETSC_COMM_WORLD,"rcmb: %g, step: %d, time: %g, erupta_total: %g, botT_val: %g\n",rcmb,step,time,erupta_total,botT_val);  

  /* 
    Allocate coordinates arrays
    Note: coordinates are only for 1 block of the yin-yang grid
    the other block can be obtained by rotation and is reconstructed later
  */
  x = (float*) malloc( ntot[0] * sizeof(float) );
  y = (float*) malloc( ntot[1] * sizeof(float) );
  z = (float*) malloc( ntot[2] * sizeof(float) );
  fread(x,sizeof(float),ntot[0],fp);
  fread(y,sizeof(float),ntot[1],fp);
  fread(z,sizeof(float),ntot[2],fp);

  if (nval != 1) { fread(&scale_fac,sizeof(float),1,fp); }

  data       = (double*) malloc( ntot[0]*ntot[1]*ntot[2]*ntot[3]*nval * sizeof(double) );
  data_local = (float*)  malloc( npi * sizeof(float) );
  /* Read blocks */
  for (ib=0; ib<npar[3]; ib++) {
    for(iz=0; iz<npar[2]; iz++) {
      for (iy=0; iy<npar[1]; iy++) {
        for (ix=0; ix<npar[0]; ix++) {
          /* Read data block */
          fread(data_local,sizeof(float),npi,fp);
          /* Copy into global array without the ghost nodes */
          for (b=0; b<ncpu[3]; b++) {
            int bb = ib*ncpu[3] + b;
            for (k=0; k<ncpu[2]; k++) {
              int kk = iz*ncpu[2] + k;
              for (j=0; j<ncpu[1]; j++) {
                int jj = iy*ncpu[1] + j;
                for (i=0; i<ncpu[0]; i++) {
                  int ii = ix*ncpu[0] + i;
                  /* 
                    Index in local array WITH ghost points
                    Layout: data_local[b][k][j][i][v] with dimensions:
                    - b: ncpu[3]
                    - k: ncpu[2]
                    - j: ncpu[1]+xyp (includes ghost)
                    - i: ncpu[0]+xyp (includes ghost)
                    - v: nval
                  */
                  int fidx_from = b * ncpu[2] * (ncpu[1]+xyp) * (ncpu[0]+xyp)
                                + k * (ncpu[1]+xyp) * (ncpu[0]+xyp)
                                + j * (ncpu[0]+xyp)
                                + i;
                  /* 
                    Index in global array WITHOUT ghost points
                    Layout: data[bb][kk][jj][ii][v] with dimensions:
                    - bb: ntot[3]
                    - kk: ntot[2]
                    - jj: ntot[1]
                    - ii: ntot[0]
                    - v: nval
                  */
                  int fidx_to = bb * ntot[2] * ntot[1] * ntot[0]
                              + kk * ntot[1] * ntot[0]
                              + jj * ntot[0]
                              + ii;
                  /* Copy all field components for this point */
                  for (v=0; v<nval; v++) {
                    int idx_from = nval*fidx_from + v;
                    int idx_to   = nval*fidx_to + v;
                    data[ idx_to ] = (double)( scale_fac * data_local[ idx_from ] );
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  free(data_local);
  fclose(fp);
  free(zg);

  /* Set output structure */
  sfield->n[0]    = ntot[0];
  sfield->n[1]    = ntot[1];
  sfield->n[2]    = ntot[2];
  sfield->nblocks = ntot[3];
  sfield->bs      = nval;
  sfield->rcmb    = rcmb;
  /* Must be freed after copied into the swarm */
  sfield->data    = data;
  sfield->x       = x;
  sfield->y       = y;
  sfield->z       = z;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static int sort_ascendant(const void *a, const void *b)
{
  int ia = *(const int*)a;
  int ib = *(const int*)b;
  if (ia < ib) return -1;
  if (ia > ib) return  1;
  return 0;
}

static PetscErrorCode ReconstructStaggGrid(StagField *sfield, PetscReal **coords, PetscReal **_fields)
{
  PetscInt  i,j,k,npoints,nblocks,bs;
  PetscInt  *n;
  PetscReal *coor,*fields;
  PetscFunctionBeginUser;
  n       = sfield->n;
  nblocks = sfield->nblocks;
  bs      = sfield->bs;
  npoints = n[0] * n[1] * n[2] * nblocks;
  PetscCall(PetscMalloc1(3*npoints,&coor));
  PetscCall(PetscMalloc1(bs*npoints,&fields));
  /* Reconstruct yin-yang grid coordinates */
  for (k=0; k<n[2]; k++) {
    PetscReal sum_P = 0.0, sum_vol = 0.0, p_avg;
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt  yin_idx,yang_idx;
        PetscReal r,lat,lon;
        PetscReal vtheta,vphi,vr;
        PetscReal p_yin,p_yang;
        
        /* Centre to 0.0 */
        r   = (PetscReal)sfield->rcmb + (PetscReal)sfield->z[k];
        lat = 0.25*PETSC_PI - (PetscReal)sfield->x[i];
        lon = (PetscReal)sfield->y[j] - 0.75*PETSC_PI;

        /* Yin */
        yin_idx = i + j*n[0] + k*n[0]*n[1];
        coor[ 3*yin_idx     ] = r * PetscCosReal(lat) * PetscCosReal(lon); /* x */
        coor[ 3*yin_idx + 1 ] = r * PetscCosReal(lat) * PetscSinReal(lon); /* y */
        coor[ 3*yin_idx + 2 ] = r * PetscSinReal(lat);                     /* z */
        
        /* velocity */
        vtheta = (PetscReal)sfield->data[ bs*yin_idx     ];
        vphi   = (PetscReal)sfield->data[ bs*yin_idx + 1 ];
        vr     = (PetscReal)sfield->data[ bs*yin_idx + 2 ];
        fields[ bs*yin_idx     ] = vtheta*PetscCosReal(0.5*PETSC_PI - lat)*PetscCosReal(lon) 
                                   - vphi*PetscSinReal(lon) 
                                   + vr*PetscSinReal(0.5*PETSC_PI - lat)*PetscCosReal(lon);
        fields[ bs*yin_idx + 1 ] = vtheta*PetscCosReal(0.5*PETSC_PI - lat)*PetscSinReal(lon)
                                   + vphi*PetscCosReal(lon) 
                                   + vr*PetscSinReal(0.5*PETSC_PI - lat)*PetscSinReal(lon);
        fields[ bs*yin_idx + 2 ] = -vtheta*PetscSinReal(0.5*PETSC_PI - lat) + vr*PetscCosReal(0.5*PETSC_PI - lat);
        /* pressure */
        p_yin = (PetscReal)sfield->data[ bs*yin_idx + 3 ];

        /* Yang */
        yang_idx = yin_idx + n[0]*n[1]*n[2];
        coor[ 3*yang_idx     ] = - coor[ 3*yin_idx     ];
        coor[ 3*yang_idx + 1 ] =   coor[ 3*yin_idx + 2 ];
        coor[ 3*yang_idx + 2 ] =   coor[ 3*yin_idx + 1 ];

        /* velocity */
        vtheta = (PetscReal)sfield->data[ bs*yang_idx     ];
        vphi   = (PetscReal)sfield->data[ bs*yang_idx + 1 ];
        vr     = (PetscReal)sfield->data[ bs*yang_idx + 2 ];
        fields[ bs*yang_idx     ] = - ( vtheta*PetscSinReal(lat)*PetscCosReal(lon)
                                      - vphi*PetscSinReal(lon)
                                      + vr*PetscCosReal(lat)*PetscCosReal(lon) );
        fields[ bs*yang_idx + 1 ] =   vtheta*PetscSinReal(lat)*PetscSinReal(lon)
                                      + vphi*PetscCosReal(lon)
                                      + vr*PetscCosReal(lat)*PetscSinReal(lon);
        fields[ bs*yang_idx + 2 ] = - vtheta*PetscCosReal(lat) + vr*PetscCosReal(0.5*PETSC_PI - lat);
        /* pressure */
        p_yang = (PetscReal)sfield->data[ bs*yang_idx + 3 ];

        sum_P   += p_yin + p_yang;
        sum_vol += 2.0;
      }
    }
    p_avg = sum_P / sum_vol;
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt  yin_idx,yang_idx;
        /* Yin */
        yin_idx = i + j*n[0] + k*n[0]*n[1];
        fields[ bs*yin_idx + 3 ] = (PetscReal)sfield->data[ bs*yin_idx + 3 ] - p_avg;
        /* Yang */
        yang_idx = yin_idx + n[0]*n[1]*n[2];
        fields[ bs*yang_idx + 3 ] = (PetscReal)sfield->data[ bs*yang_idx + 3 ] - p_avg;
      }
    }
  }
  *coords = coor;
  *_fields = fields;
  free(sfield->data);
  free(sfield->x);
  free(sfield->y);
  free(sfield->z);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RemoveYinYangCorners(DM swarm, StagField *sfield)
{
  PetscInt  i,j,k,np_yin,cnt;
  PetscInt  *n,*rm_idx;
  PetscReal *coords;
  PetscFunctionBeginUser;

  n      = sfield->n;
  np_yin = n[0] * n[1] * n[2];
  PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  PetscCall(PetscMalloc1(np_yin,&rm_idx));
  cnt = 0;
#if 1
  for (k=0; k<n[2]; k++) {
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt yin_idx,yang_idx;
        PetscReal x1,y1,z1;
        PetscReal x2,y2,z2;
        PetscReal theta1,phi1;
        PetscReal theta2,phi2;

        /* Yin grid */
        yin_idx  = i + j*n[0] + k*n[0]*n[1];
        /* Yang grid */
        yang_idx = yin_idx + n[0]*n[1]*n[2];

        x1 = coords[ 3*yin_idx     ];
        y1 = coords[ 3*yin_idx + 1 ];
        z1 = coords[ 3*yin_idx + 2 ];

        x2 = coords[ 3*yang_idx     ];
        y2 = coords[ 3*yang_idx + 1 ];
        z2 = coords[ 3*yang_idx + 2 ];

        theta1 = PetscAtan2Real( PetscSqrtReal(x1*x1 + y1*y1), z1 );
        phi1   = PetscAtan2Real( y1, x1 );

        theta2 = PetscAtan2Real( PetscSqrtReal(x2*x2 + y2*y2), z2 );
        phi2   = PetscAtan2Real( y2, x2 );
        
        if ( ( theta2 > 0.25*PETSC_PI  && theta2 < 0.75*PETSC_PI && 
               phi2   > -0.75*PETSC_PI && phi2   < 0.75*PETSC_PI ) ) {
          if (x1 < 0.0) rm_idx[cnt++] = yin_idx;
          if (x2 > 0.0) rm_idx[cnt++] = yang_idx;
        }
        
      }
    }
  }
#endif
  PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  if (cnt == 0) {
    PetscFree(rm_idx);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Remove marked points */
  //PetscLogDouble time[2];
  //PetscCall(PetscTime(&time[0]));
  //qsort(rm_idx,cnt,sizeof(PetscInt),sort_ascendant);
  //PetscCall(PetscIntSortSemiOrdered(cnt, rm_idx));
  //PetscCall(PetscSortInt(cnt, rm_idx));
  //PetscCall(PetscTime(&time[1]));
  //PetscPrintf(PETSC_COMM_WORLD,"Sorting %" PetscInt_FMT " indices took %g sec\n",cnt,time[1]-time[0]);
  PetscCall(DMSwarmRemoveNPointsFromIndex(swarm, cnt, rm_idx));
  //for (i=0; i<cnt; i++) {
  //  PetscCall(DMSwarmRemovePointAtIndex(swarm, rm_idx[i]));
  //}
  PetscFree(rm_idx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM *sw)
{
  DM swarm;
  PetscInt dim = 3;
  char     dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD,&swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_BASIC)); // Will use DMSwarmMigrate_Push_Basic
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));

  PetscCall(DMSwarmInitializeFieldRegister(swarm)); // this register by default the fields DMSwarmField_pid & DMSwarmField_rank

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,DMSwarmPICField_coor,dim,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",4,PETSC_REAL));


  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));
  *sw = swarm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode TestStagBinMain()
{
  LagRBFCtx ctx;
  StagField sfield;
  DM        swarm;
  PetscInt  npoints;
  PetscReal *coor,*coords,*fields,*data;
  PetscBool exists;
  const char *output_fields[] = { DMSwarmPICField_coor, "fields" };
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }

  PetscCall(PetscMemzero(&sfield,sizeof(StagField)));
  PetscCall(TestBinaryStagg(&sfield));
#if 1
  PetscCall(ReconstructStaggGrid(&sfield,&coor,&data));
  PetscCall(CreateSwarm(&swarm));

  npoints = sfield.n[0] * sfield.n[1] * sfield.n[2] * sfield.nblocks;
  PetscCall(DMSwarmSetLocalSizes(swarm,npoints,-1));
  PetscPrintf(PETSC_COMM_WORLD,"Stag: npoints = %d x %d x %d x %d = %d \n",sfield.n[0],sfield.n[1],sfield.n[2],sfield.nblocks,npoints);
  PetscPrintf(PETSC_COMM_WORLD,"Stag: bs = %d \n",sfield.bs);

  /* Set coords and velocity fields */
  //PetscCall(DMSwarmGetField(swarm,"coords",NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmGetField(swarm,"fields",NULL,NULL,(void**)&fields));
  PetscCall(PetscArraycpy(coords,coor,3*npoints));
  PetscCall(PetscArraycpy(fields,data,4*npoints));
  //PetscCall(DMSwarmRestoreField(swarm,"coords",NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmRestoreField(swarm,"fields",NULL,NULL,(void**)&fields));

  PetscFree(coor);
  PetscFree(data);
  PetscCall(RemoveYinYangCorners(swarm,&sfield));

  PetscCall(SwarmOutput(ctx,swarm,2,output_fields,PETSC_FALSE));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if 0
#include "readers/stagYY/binary_loader.h"
static PetscErrorCode TestStagBinMain()
{
  LagRBFCtx ctx;
  DM        swarm;
  PetscBool exists;
  const char *output_fields[] = { DMSwarmPICField_coor, "fields" };
  const char *fname[] = { "/mnt/c/Users/jourdon/Documents/Scripts/PostProc/StagYY/PJB6_YS1_Rh32_vp00039" };
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(TestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(ctx->output_dir)); }

  PetscCall(CreateSwarm(&swarm));
  PetscCall(StagBinSwarmSetFromFiles(swarm, fname, 1, PETSC_FALSE));
  PetscCall(SwarmOutput(ctx,swarm,2,output_fields,PETSC_FALSE));

  PetscCall(DMDestroy(&swarm));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif


static PetscErrorCode CountZerosInRBFMatrix(LagRBFCtx ctx, DM swarm)
{
  DMSwarmCellDM    celldm;
  PartitionOfUnity pu;
  Partition       *partitions;
  PetscInt         n,dim,npart;
  const char       **coordFieldsName;
  PetscReal        *coords;
  PetscReal        epsilon;
  PetscMPIInt      rank;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  PetscCall(PUGetEpsilon(pu,&epsilon));
  
  PetscCall(DMSwarmSortGetAccess(swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFieldsName));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm, coordFieldsName[0], NULL, NULL, (void **)&coords));

  for (n=0; n<npart; n++) {
    Partition  p = partitions[n];
    PetscInt64 zero_count = 0;
    PetscInt   npoints,i,j,d;
    PetscInt   *pidx;
    PetscReal  ratio;

    PetscCall(PartitionFillDataPointsIdx(p,swarm));
    PetscCall(PartitionGetDataPointsIdx(p,&npoints,&pidx));
    if (npoints == 0) continue;
    /* RBF matrix */
    for (i=0; i<npoints; i++) {
      for (j=i; j<npoints; j++) {
        PetscReal dist = 0.0;
        /* Compute the distance between points */
        for (d=0; d<dim; d++) {
          dist += (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]) * (coords[ dim*pidx[i] + d] - coords[ dim*pidx[j] + d]);
        }
        dist = PetscSqrtReal(dist);
        if (1.0 - epsilon * dist < 0.0) {
          zero_count++;
        }
      }
    }
    ratio = (PetscReal)(2*zero_count) / (PetscReal)(npoints*npoints);
    PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: Partition[%" PetscInt_FMT "] has %" PetscInt_FMT " data points and %" PetscInt64_FMT " zero entries = %g ratio, in the RBF matrix\n",rank,n,npoints,2*zero_count,ratio);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#include "readers/stagYY/binary_loader.h"
static PetscErrorCode TestStagBinMain()
{
  PartitionOfUnity pu;
  LagRBFCtx ctx;
  Partition *partitions;
  CellInfo  *cellinfo;
  DM        swarm_data,swarm_interp;
  PetscInt  np,cnt,npoints_i,npoints_d,output_frequency,step,nsteps;
  PetscInt  *rm_idx;
  PetscReal time,time_final,time_last_output,output_dt,dt;
  PetscBool exists;
  PetscMPIInt rank;
  //const char *output_fields[] = { DMSwarmPICField_coor, "fields" };
  const char *fname[] = { "/home/jourdon/data/LagRBF_input/PJB6_YS1_Rh32_vp00039" };
  const char *output_fields[] = {"DMSwarm_pid","fields","velocity","DMSwarm_rank"};
  PetscFunctionBeginUser;

  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));

  PetscCall(PetscTestDirectory(ctx->output_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->output_dir); }
  PetscCall(PetscTestDirectory(ctx->mat_dir,'w',&exists));
  if (!exists) { CreateDirectory(ctx->mat_dir); }

  /* Create data structures */
  ctx->mtype = GEODYNAMICS; // Set the model type
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));
  PetscCall(LagRBF_CellInfoCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetCellInfo(ctx,&cellinfo));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  PetscCall(PartitionOfUnityMeshView(pu));
#if 1
  PetscCall(SwarmDataCreate(ctx,&swarm_data));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  /* Read data only on rank 0 */
  if (rank == 0) {
    PetscCall(StagBinSwarmSetFromFiles(swarm_data, fname, 1, PETSC_FALSE));
  }
  /* Send data to other ranks */
  PetscCall(SwarmDataAssignMigrateBasic(ctx,swarm_data));
  /* Duplicate ghost points */
  PetscCall(SwarmDataAssignMigrateDuplicateGhostPoints(ctx,swarm_data));
  /* Set cell IDs */
  PetscCall(DMSwarmPICSetCellId(swarm_data));
  ctx->swd = swarm_data;

  //PetscCall(CountZerosInRBFMatrix(ctx, swarm_data));

  PetscCall(DMSwarmGetLocalSize(swarm_data,&npoints_d));
  //PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: swarm data contains: %" PetscInt_FMT " points\n",rank,npoints_d);

  /* Set up CellInfo data structure */
  PetscCall(CellInfoSetUp(pu,cellinfo,swarm_data));
  /* Create swarm for the interpolation */
  PetscCall(SwarmInterpSetUp(ctx,&swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&npoints_i));
  PetscCall(SwarmFieldAssignUniqueIndex(swarm_interp,DMSwarmField_pid,0,npoints_i));
  ctx->swi = swarm_interp;
  //PetscPrintf(PETSC_COMM_SELF,"Rank[%d]: swarm interp contains: %" PetscInt_FMT " points\n",rank,npoints_i);
#if 1
  /* Set up Partition data structure */
  //PetscPrintf(PETSC_COMM_WORLD,"Setting up partitions...\n");
  PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
  /* Interpolate */
  //PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
  PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
#endif
  //PetscCall(SwarmOutput(ctx,swarm_data,2,output_fields,PETSC_FALSE));
  PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
#if 0
  /* Migrate test */
  for (step=1; step<5; step++){
    DMSwarmCellDM  celldm;
    DM       dmda;
    PetscInt npp;
    long int np_before_local, np_before_global;
    long int np_after_local,  np_after_global;
    const char *dmda_name;

    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"=================== MIGRATION STEP %d ===================\n",step));
    PetscCall(DMSwarmGetLocalSize(swarm_interp,&npp));
    np_before_local = (long int)npp;
    PetscCallMPI(MPI_Allreduce(&np_before_local,&np_before_global,1,MPI_LONG,MPI_SUM,PETSC_COMM_WORLD));

    PetscCall(DMSwarmGetCellDMActive(swarm_interp,&celldm));
    PetscCall(DMSwarmCellDMGetDM(celldm,&dmda));
    PetscCall(PetscObjectGetName((PetscObject)dmda,&dmda_name));
    PetscPrintf(PETSC_COMM_WORLD,"[[ Migration ]]: Using cell dm %s\n",dmda_name);

  PetscCall(DMSwarmMigrate(swarm_interp,PETSC_TRUE)); // update cell ids and MPI ranks

    PetscCall(DMSwarmGetLocalSize(swarm_interp,&npp));
    np_after_local = (long int)npp;
    PetscCallMPI(MPI_Allreduce(&np_after_local,&np_after_global,1,MPI_LONG,MPI_SUM,PETSC_COMM_WORLD));
    PetscPrintf(PETSC_COMM_WORLD,"[[ Migration ]]: Stage %d: Migrated points: before = %ld, after = %ld\n",step,np_before_global,np_after_global);

    PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
  }
#endif
#if 1
  /* Advection */
  /* Copy the first dim fields into "velocity" */
  PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
  /* Output */
  PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
  output_frequency = 100;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-output_frequency",&output_frequency,NULL));
  time_last_output = 0.0;
  output_dt = 0.1;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-output_dt",&output_dt,NULL));
  dt = 1.0e-2;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL));
  time_final = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-time_final",&time_final,NULL));
  nsteps = 10000;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsteps",&nsteps,NULL));
  time = 0.0;
  PetscCall(LagRBFSetTime(ctx,time));
  PetscCall(LagRBFSetTimeStep(ctx,dt));
  step = 1;
  while (time < time_final) {
    PetscCall(LagRBFSetStep(ctx,step));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[[ STEP: %" PetscInt_FMT " ]], time: %g\n",step,time));
    /* Interpolate */
    PetscCall(LagRBF_PartitionsSetUp(ctx,swarm_data,swarm_interp));
    //PetscCall(PartitionSolveInterpolationCoefficients(ctx,swarm_data,"fields"));
    PetscCall(RBFInterpolation(ctx,swarm_interp,swarm_data,"fields"));
    PetscCall(AdvectionInitializeTimestepVelocity(ctx,swarm_interp));
    /* Output */
    if (step % output_frequency == 0) {
      PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
    }
    /* Output every given dt_output */
    if (output_dt > 0.0) {
      if (time - time_last_output >= output_dt - 1e-10) {
        PetscCall(SwarmOutput(ctx,swarm_interp,4,output_fields,PETSC_FALSE));
        time_last_output = time;
      }
    }
    /* Advect */
    PetscCall(LagRBF_AdvectSwarm(ctx,swarm_interp,swarm_data));
    PetscCall(LagRBFGetTimeStep(ctx,&dt));
    if (time + dt > time_final) {
      dt = time_final - time;
      PetscCall(LagRBFSetTimeStep(ctx,dt));
    }
    time += dt;
    PetscCall(LagRBFSetTime(ctx,time));
    step++;
    if (step > nsteps) { break; }
  }
  /* 
  For now I only make tests on a single stag time step,
  but when other stag time step will be taken into account 
  I will need to re-solve the partitions for each stag time step (but not for each advection step)
  */
  PetscCall(LagRBF_PartitionsReset(ctx));
#endif
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
#endif
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(LagRBFInitialize(&argc,&argv,0,help));

  //PetscCall(Test());
  PetscCall(TestStagBinMain());

  PetscCall(PetscFinalize());
  
  return 0;
}