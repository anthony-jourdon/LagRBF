#include <petsc.h>
#include "readers/stagYY/binary_loader.h"
#include "swarm/swarm_utils.h"

#define DEBUG

static PetscErrorCode StagFieldBinFreeArrays(StagFieldBin *field)
{
  PetscFunctionBeginUser;
  if (field->data) PetscFree(field->data);
  field->data = NULL;
  if (field->x)    PetscFree(field->x);
  field->x = NULL;
  if (field->y)    PetscFree(field->y);
  field->y = NULL;
  if (field->z)    PetscFree(field->z);
  field->z = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagReadBinaryFile(const char *filename, StagFieldBin *sfield)
{
  FILE       *fp;
  PetscInt   nval;
  int        i,j,k,ib,iz,iy,ix,b,v;
  int        magic,xyp,nz2,npi,step;
  int        ncpu[4],ntot[4],npar[4];
  float      rcmb,time,erupta_total,botT_val,scale_fac=1.0;
  float      aspect[2];
  float      *zg,*data_local;
  double     *data;
  PetscFunctionBeginUser;
  
  nval = sfield->bs;
  /* Can only be scalar => nval = 1 or velocity-pressure => nval = 4 */
  PetscCheck(nval == 1 || nval == 4,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block-size bs must be 1 (scalar) or 4 (velocity-pressure), found bs = %"PetscInt_FMT,nval);

  fp = fopen(filename,"rb");
  PetscCheck(fp != NULL,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file: %s",filename);

  fread(&magic,sizeof(int),1,fp);
  if ( (magic < 100 && nval > 1) || (magic > 300 && nval == 1) ) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"File %s contains the wrong number of components in provided field.",filename);
  }
  magic = magic % 100;
  if (magic >= 9 && nval == 4) { xyp = 1; } // extra ghost point in x & y directions
  else                         { xyp = 0; }

  fread(ntot,sizeof(int),4,fp);     /* Total number of grid points + # blocks (2 for yin-yang grid) */
  fread(aspect,sizeof(float),2,fp); /* Aspect ratios */
  fread(npar,sizeof(int),4,fp);     /* Number of parallel subdomains */
  // z-coordinates of what ??
  nz2 = 2*ntot[2] + 1;
  PetscCall(PetscMalloc1(nz2,&zg)); //(float*) malloc( nz2 * sizeof(float) );
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

#ifdef DEBUG
  // Report for debugging
  PetscPrintf(PETSC_COMM_WORLD,"nval: %d\n",nval);
  PetscPrintf(PETSC_COMM_WORLD,"Magic modulo 100: %d\n",magic);
  PetscPrintf(PETSC_COMM_WORLD,"ntot: %d %d %d %d\n",ntot[0],ntot[1],ntot[2],ntot[3]);
  PetscPrintf(PETSC_COMM_WORLD,"Aspect ratios: %g %g\n",aspect[0],aspect[1]);
  PetscPrintf(PETSC_COMM_WORLD,"npar: %d %d %d %d\n",npar[0],npar[1],npar[2],npar[3]);
  PetscPrintf(PETSC_COMM_WORLD,"n per CPU: %d %d %d %d\n",ncpu[0],ncpu[1],ncpu[2],ncpu[3]);
  PetscPrintf(PETSC_COMM_WORLD,"Number of values per block: %d\n",npi);
  PetscPrintf(PETSC_COMM_WORLD,"rcmb: %g, step: %d, time: %g, erupta_total: %g, botT_val: %g\n",rcmb,step,time,erupta_total,botT_val);  
#endif
  /* 
    Allocate coordinates arrays
    Note: coordinates are only for 1 block of the yin-yang grid
    the other block can be obtained by rotation and is reconstructed later
  */
  if (!sfield->x) { PetscCall(PetscMalloc1(ntot[0],&sfield->x)); }
  if (!sfield->y) { PetscCall(PetscMalloc1(ntot[1],&sfield->y)); }
  if (!sfield->z) { PetscCall(PetscMalloc1(ntot[2],&sfield->z)); }
  fread(sfield->x,sizeof(float),ntot[0],fp);
  fread(sfield->y,sizeof(float),ntot[1],fp);
  fread(sfield->z,sizeof(float),ntot[2],fp);

  if (nval != 1) { fread(&scale_fac,sizeof(float),1,fp); }
  PetscCall(PetscMalloc1(ntot[0]*ntot[1]*ntot[2]*ntot[3]*nval,&data));
  PetscCall(PetscMalloc1(npi,&data_local));
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
  PetscCall(PetscFree(data_local));
  PetscCall(PetscFree(zg));
  fclose(fp);

  /* Set field data */
  sfield->n[0]    = ntot[0];
  sfield->n[1]    = ntot[1];
  sfield->n[2]    = ntot[2];
  sfield->nblocks = ntot[3];
  sfield->rcmb    = rcmb;
  sfield->data    = data;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode Cartesian2Spherical(PetscReal cartesian[3], PetscReal spherical[3])
{
  PetscFunctionBeginUser;
  spherical[0] = PetscSqrtReal( cartesian[0]*cartesian[0] + cartesian[1]*cartesian[1] + cartesian[2]*cartesian[2] ); // r
  spherical[1] = PetscAtan2Real( PetscSqrtReal(cartesian[0]*cartesian[0] + cartesian[1]*cartesian[1]), cartesian[2] ); // theta -> colatitude
  spherical[2] = PetscAtan2Real( cartesian[1], cartesian[0] ); // phi -> longitude
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode Spherical2Cartesian(PetscReal spherical[3], PetscReal cartesian[3])
{
  PetscFunctionBeginUser;
  cartesian[0] = spherical[0] * PetscCosReal(spherical[1]) * PetscCosReal(spherical[2]); // x
  cartesian[1] = spherical[0] * PetscCosReal(spherical[1]) * PetscSinReal(spherical[2]); // y
  cartesian[2] = spherical[0] * PetscSinReal(spherical[1]);                              // z
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReconstructYinYangVelocityPressure(StagFieldBin *sfield, PetscReal *coor, PetscInt nfields, PetscReal *fields)
{
  PetscInt  i,j,k,bs;
  PetscInt  *n;
  PetscFunctionBeginUser;
  n       = sfield->n;
  bs      = sfield->bs;
  PetscCheck(bs == 4,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block-size bs must be 4 for velocity-pressure fields. Found bs = %"PetscInt_FMT,bs);
  /* Reconstruct yin-yang grid coordinates */
  for (k=0; k<n[2]; k++) {
    PetscReal sum_P = 0.0, sum_vol = 0.0, p_avg;
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt  yin_idx,yang_idx;
        PetscReal p_yin,p_yang;
        PetscReal vr,vtheta,vphi;
        PetscReal spherical[3];
        
        yin_idx  = i + j*n[0] + k*n[0]*n[1];
        yang_idx = yin_idx + n[0]*n[1]*n[2];

        /* coordinates */
        spherical[0] = (PetscReal)sfield->rcmb + (PetscReal)sfield->z[k]; // r
        spherical[1] = 0.25*PETSC_PI - (PetscReal)sfield->x[i];           // theta -> colatitude
        spherical[2] = (PetscReal)sfield->y[j] - 0.75*PETSC_PI;           // phi -> longitude

        /* Yin grid */
        PetscCall(Spherical2Cartesian(spherical, &coor[ 3*yin_idx ]));
        /* Yang grid */
        coor[ 3*yang_idx     ] = - coor[ 3*yin_idx     ];
        coor[ 3*yang_idx + 1 ] =   coor[ 3*yin_idx + 2 ];
        coor[ 3*yang_idx + 2 ] =   coor[ 3*yin_idx + 1 ];

        /* velocity */
        /* Yin grid */
        vtheta = (PetscReal)sfield->data[ bs*yin_idx     ];
        vphi   = (PetscReal)sfield->data[ bs*yin_idx + 1 ];
        vr     = (PetscReal)sfield->data[ bs*yin_idx + 2 ];
        fields[ nfields*yin_idx     ] = vtheta * PetscSinReal(spherical[1]) * PetscCosReal(spherical[2])
                                      - vphi   * PetscSinReal(spherical[2])
                                      + vr     * PetscCosReal(spherical[1]) * PetscCosReal(spherical[2]);
        fields[ nfields*yin_idx + 1 ] = vtheta * PetscSinReal(spherical[1]) * PetscSinReal(spherical[2])
                                      + vphi   * PetscCosReal(spherical[2])
                                      + vr     * PetscCosReal(spherical[1]) * PetscSinReal(spherical[2]);
        fields[ nfields*yin_idx + 2 ] = - vtheta * PetscCosReal(spherical[1])
                                      + vr       * PetscSinReal(spherical[1]);
        /* Yang grid */
        vtheta = (PetscReal)sfield->data[ bs*yang_idx     ];
        vphi   = (PetscReal)sfield->data[ bs*yang_idx + 1 ];
        vr     = (PetscReal)sfield->data[ bs*yang_idx + 2 ];
        
        fields[ nfields*yang_idx     ] = - vtheta * PetscSinReal(spherical[1]) * PetscCosReal(spherical[2]) 
                                    + vphi   * PetscSinReal(spherical[2])
                                    - vr     * PetscCosReal(spherical[1]) * PetscCosReal(spherical[2]);
        fields[ nfields*yang_idx + 2 ] =   vtheta * PetscSinReal(spherical[1]) * PetscSinReal(spherical[2])
                                    + vphi   * PetscCosReal(spherical[2])
                                    + vr     * PetscCosReal(spherical[1]) * PetscSinReal(spherical[2]);
        fields[ nfields*yang_idx + 1 ] = - vtheta * PetscCosReal(spherical[1])
                                    + vr     * PetscSinReal(spherical[1]);
        
        /* pressure */
        p_yin  = (PetscReal)sfield->data[ bs*yin_idx + 3 ];
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
        fields[ nfields*yin_idx + 3 ] = (PetscReal)sfield->data[ bs*yin_idx + 3 ] - p_avg;
        /* Yang */
        yang_idx = yin_idx + n[0]*n[1]*n[2];
        fields[ nfields*yang_idx + 3 ] = (PetscReal)sfield->data[ bs*yang_idx + 3 ] - p_avg;
      }
    }
  }
  sfield->yy_reconstructed = true;
  PetscCall(StagFieldBinFreeArrays(sfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReconstructYinYangGrid(StagFieldBin *sfield, PetscReal *coor)
{
  PetscInt  i,j,k;
  PetscInt  *n;
  PetscFunctionBeginUser;
  n       = sfield->n;
  /* Reconstruct yin-yang grid coordinates */
  for (k=0; k<n[2]; k++) {
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt  yin_idx,yang_idx;
        PetscReal spherical[3];
        
        yin_idx  = i + j*n[0] + k*n[0]*n[1];
        yang_idx = yin_idx + n[0]*n[1]*n[2];

        /* coordinates */
        spherical[0] = (PetscReal)sfield->rcmb + (PetscReal)sfield->z[k]; // r
        spherical[1] = 0.25*PETSC_PI - (PetscReal)sfield->x[i];           // theta -> colatitude
        spherical[2] = (PetscReal)sfield->y[j] - 0.75*PETSC_PI;           // phi -> longitude

        /* Yin grid */
        PetscCall(Spherical2Cartesian(spherical, &coor[ 3*yin_idx ]));
        /* Yang grid */
        coor[ 3*yang_idx     ] = - coor[ 3*yin_idx     ];
        coor[ 3*yang_idx + 1 ] =   coor[ 3*yin_idx + 2 ];
        coor[ 3*yang_idx + 2 ] =   coor[ 3*yin_idx + 1 ];
      }
    }
  }
  PetscCall(PetscFree(sfield->x));
  PetscCall(PetscFree(sfield->y));
  PetscCall(PetscFree(sfield->z));
  sfield->yy_reconstructed = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagInsertScalarFieldData(StagFieldBin *sfield, PetscInt block, PetscInt bs, PetscReal *fields)
{
  PetscInt i,j,k;
  PetscInt *n;
  PetscFunctionBeginUser;
  n = sfield->n;
  for (k=0; k<n[2]; k++) {
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt yin_idx,yang_idx;
        yin_idx  = i + j*n[0] + k*n[0]*n[1];
        yang_idx = yin_idx + n[0]*n[1]*n[2];
        fields[ bs*yin_idx + block ]  = (PetscReal)sfield->data[ yin_idx ];
        fields[ bs*yang_idx + block ] = (PetscReal)sfield->data[ yang_idx ];
      }
    }
  }
  PetscCall(PetscFree(sfield->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagBinSwarmRemoveYinYangCorners(DM swarm, StagFieldBin *sfield)
{
  PetscInt i,j,k,cnt,half_np;
  PetscInt *n,*rm_idx;
  PetscReal *coords;
  PetscFunctionBeginUser;
  n       = sfield->n;
  half_np = n[0] * n[1] * n[2];
  PetscCall(PetscMalloc1(2*half_np,&rm_idx));
  PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  cnt = 0; // initialize counter of points to remove
  for (k=0; k<n[2]; k++) {
    for (j=0; j<n[1]; j++) {
      for (i=0; i<n[0]; i++) {
        PetscInt  yin_idx,yang_idx;
        PetscReal x_yin,x_yang,y_yang,z_yang,theta_yang,phi_yang;

        yin_idx  = i + j*n[0] + k*n[0]*n[1];
        yang_idx = yin_idx + half_np;
        /* Yin x Cartesian coordinate */
        x_yin  = coords[ 3*yin_idx     ];
        /* Yang Cartesian coordinates */
        x_yang = coords[ 3*yang_idx     ];
        y_yang = coords[ 3*yang_idx + 1 ];
        z_yang = coords[ 3*yang_idx + 2 ];
        /* Yang colatitude and longitude */
        theta_yang = PetscAtan2Real( PetscSqrtReal(x_yang*x_yang + y_yang*y_yang), z_yang );
        phi_yang   = PetscAtan2Real( y_yang, x_yang );
        /* Check if point is in the overlapping region */
        if ( ( theta_yang > 0.25*PETSC_PI  && theta_yang < 0.75*PETSC_PI && 
               phi_yang   > -0.75*PETSC_PI && phi_yang   < 0.75*PETSC_PI ) ) {
          if (x_yin  < 0.0) rm_idx[cnt++] = yin_idx;
          if (x_yang > 0.0) rm_idx[cnt++] = yang_idx;
        }
      }
    }
  }
  PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  /* If no points to remove, return */
  if (cnt == 0) {
    PetscCall(PetscFree(rm_idx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Remove points */
  PetscCall(DMSwarmRemoveNPointsFromIndex(swarm, cnt, rm_idx));
  PetscCall(PetscFree(rm_idx));
  /* mark that the yin-yang grid has been reconstructed and corners removed */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagBinSwarmSetVelocityPressureFromFile(DM swarm, const char *filename, StagFieldBin *sfield)
{
  PetscInt  npoints,bs;
  PetscReal *coords = NULL, *fields = NULL;
  PetscFunctionBeginUser;

  sfield->bs = 4; // velocity + pressure
  /* Read binary file and store data in sfield */
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(StagReadBinaryFile(filename,sfield));
  if (npoints <= 0) {
    PetscInt np;
    PetscInt *n;

    n  = sfield->n;
    np = n[0] * n[1] * n[2] * sfield->nblocks;
    PetscCall(DMSwarmSetLocalSizes(swarm,np,-1));
    /* If swarm is resized, mark that the yin-yang grid has not been reconstructed */
    sfield->yy_reconstructed = false; 
  }
  /* Access coordinates and fields */
  PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmGetField(swarm,"fields",&bs,NULL,(void**)&fields));
  /* Set OVERLAPPING yin-yang coordinates and velocity-pressure fields */
  PetscCall(ReconstructYinYangVelocityPressure(sfield,coords,bs,fields));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  PetscCall(DMSwarmRestoreField(swarm,"fields",&bs,NULL,(void**)&fields));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagBinSwarmSetScalarFieldsFromFile(DM swarm, const char *fname, PetscInt block, StagFieldBin *sfield)
{
  PetscInt  bs,npoints;
  PetscReal *coords = NULL, *fields = NULL;
  PetscFunctionBeginUser;

  sfield->bs = 1; // scalar field
  PetscCall(StagReadBinaryFile(fname,sfield));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  /* If there are no points => resize the swarm */
  if (npoints <= 0) {
    PetscInt np;
    PetscInt *n;

    n  = sfield->n;
    np = n[0] * n[1] * n[2] * sfield->nblocks;
    PetscCall(DMSwarmSetLocalSizes(swarm,np,-1));
    /* If swarm is resized, mark that the yin-yang grid has not been reconstructed */
    sfield->yy_reconstructed = false; 
  }
  /* Fields */
  PetscCall(DMSwarmGetField(swarm,"fields", &bs, NULL, (void**)&fields));
  PetscCheck(bs > block,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block index for scalar field (%"PetscInt_FMT") exceeds number of swarm fields (%"PetscInt_FMT")",block,bs);
  PetscCall(StagInsertScalarFieldData(sfield, block, bs, fields));
  PetscCall(DMSwarmRestoreField(swarm,"fields", &bs, NULL, (void**)&fields));

  /* Coordinates */
  if (!sfield->yy_reconstructed) {
    PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
    /* Reconstruct yin-yang grid coordinates */
    PetscCall(ReconstructYinYangGrid(sfield,coords));
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&coords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StagBinSwarmSetFromFiles(DM swarm, char **files_list, PetscInt nfiles, PetscBool *scalars)
{
  StagFieldBin sfield;
  PetscInt     f,block;
  PetscFunctionBeginUser;

  if (nfiles < 1) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"At least one file must be provided."); }

  /* Initialize StagFieldBin */
  PetscCall(PetscMemzero(&sfield,sizeof(StagFieldBin)));
  /* 
  If there is the velocity/pressure file (scalar[f] is false) we need to ensure it will be the first
  entries in the swarm for the advection stuff. 
  */
  for (f=0; f<nfiles; f++) {
    if (!scalars[f]) {
      if (f != 0) {
        /* Swap files_list[0] and files_list[f] */
        char *temp    = files_list[0];
        files_list[0] = files_list[f];
        files_list[f] = temp;
        /* Swap scalars[0] and scalars[f] */
        PetscBool temp_bool = scalars[0];
        scalars[0] = scalars[f];
        scalars[f] = temp_bool;
      }
      break;
    }
  }
  block = 0;
  for (f=0; f<nfiles; f++) {
    PetscPrintf(PETSC_COMM_WORLD,"Loading data from file %s in block %" PetscInt_FMT "\n",files_list[f],block);
    if (scalars[f]) {
      StagBinSwarmSetScalarFieldsFromFile(swarm,files_list[f],block,&sfield);
      block++;
    } else {
      StagBinSwarmSetVelocityPressureFromFile(swarm,files_list[f],&sfield);
      block += 4; // velocity + pressure
    }
  }
  /* Remove points in overlapping region */
  PetscCall(StagBinSwarmRemoveYinYangCorners(swarm,&sfield));
  /* Ensure everything has been freed */
  PetscCall(StagFieldBinFreeArrays(&sfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}