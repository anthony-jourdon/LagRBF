#include <petsc.h>
#include "lagrangian_rbf_init.h"
#include "lagrangian_rbf.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"
#include "swarm_utils.h"
#include "swarm_interp.h"
#include "swarm_data.h"
#include "interpolation.h"

#include <immintrin.h>

static double franke_function(double x, double y)
{
  /* Franke's function */
  /* f1 = 0.75 * np.exp(-((9 * x - 2)**2 + (9 * y - 2)**2) / 4)
     f2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49 - (9 * y + 1)**2 / 10)
     f3 = 0.5 * np.exp(-((9 * x - 7)**2 + (9 * y - 3)**2) / 4)
     f4 = 0.2 * np.exp(-((9 * x - 4)**2 + (9 * y - 7)**2))
     f = f1 + f2 + f3 - f4 
   */
  double value;
  
  value  =  0.75 * PetscExpReal(-((9.0 * x - 2.0) * (9.0 * x - 2.0) + (9.0 * y - 2.0) * (9.0 * y - 2.0)) / 4.0);
  value +=  0.75 * PetscExpReal(-((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0 - (9.0 * y + 1.0) * (9.0 * y + 1.0) / 10.0);
  value +=  0.5 * PetscExpReal(-((9.0 * x - 7.0) * (9.0 * x - 7.0) + (9.0 * y - 3.0) * (9.0 * y - 3.0)) / 4.0);
  value += -0.2 * PetscExpReal(-((9.0 * x - 4.0) * (9.0 * x - 4.0) + (9.0 * y - 7.0) * (9.0 * y - 7.0)));

  return value;
}

static PetscReal triple_gaussians(PetscReal x, PetscReal y)
{
  /*
    0.81*exp(-0.9*(x + 1.0)**2 - 0.9*(z + 1.0)**2)
    1.0*exp(-4.0*x**2 - 4.0*(z - 1.5)**2)
    0.5625*exp(-0.75*z**2 - 0.75*(x - 1.5)**2)
  */
  PetscReal value;
  PetscFunctionBeginUser;
  
  value = 0.81*PetscExpReal(-0.9*((x + 1.0)*(x + 1.0) + (y + 1.0)*(y + 1.0)));
  value += 1.0*PetscExpReal(-4.0*(x*x + (y - 1.5)*(y - 1.5)));
  value += 0.5625*PetscExpReal(-0.75*((x - 1.5)*(x - 1.5) + y*y));
  PetscFunctionReturn(value);
}

static PetscReal circle_gaussian(PetscReal x, PetscReal y)
{
  PetscReal value;
  PetscFunctionBeginUser;
  value = 0.5*PetscExpReal(-3.0*pow(PetscSqrtReal(x*x + y*y) - 1.0, 2));
  PetscFunctionReturn(value);
}

static PetscReal discontinuous_cross_planes(PetscReal x, PetscReal y)
{
  PetscReal sqrt_2_o_2 = 0.70710678118654757; // sqrt(2)/2
  PetscFunctionBeginUser;
  if (sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
    if (-sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
      PetscFunctionReturn(1.0); // first quadrant
    } else {
      PetscFunctionReturn(2.0); // second quadrant
    }
  } else {
    if (-sqrt_2_o_2 * x + sqrt_2_o_2 * y < 0.0) {
      PetscFunctionReturn(3.0); // third quadrant
    } else {
      PetscFunctionReturn(4.0); // fourth quadrant
    }
  }
}

static PetscErrorCode CreateDataPoints(LagRBFCtx ctx, DM *sw)
{
  DM         swarm;
  PetscInt   dim,nn;
  PetscInt   *m;
  PetscReal  O[3],L[3];
  PetscBool  found;
  char       option_name[PETSC_MAX_PATH_LEN];
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(SwarmDataCreate(ctx,&swarm));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(PetscMalloc1(dim,&m));

  PetscCall(DMGetOptionsPrefix(swarm,&prefix));
  PetscCall(PetscSNPrintf(option_name,PETSC_MAX_PATH_LEN-1,"-%sn",prefix));
  nn = dim;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,option_name,m,&nn,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%s not found. User must provide the number of data points in each dimension.",option_name);
  PetscCheck(nn == dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%s must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",option_name,dim,nn);

  PetscCall(DMGetBoundingBox(ctx->pu->dap,O,L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscCall(PetscFree(m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmDataSetValues(DM sw)
{
  DMSwarmCellDM celldm;
  PetscInt      p,bs,dim,npoints;
  PetscReal     *coords,*fields;
  const char    **coorFieldName;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMGetDimension(sw,&dim));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(DMSwarmGetField(sw, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,"fields",&bs,NULL,(void **)&fields));

  for (p=0; p<npoints; p++) {
    if (bs == 3) {
      
      fields[bs*p    ] = triple_gaussians(coords[dim*p], coords[dim*p+1]); // first field
      fields[bs*p + 1] = circle_gaussian(coords[dim*p], coords[dim*p+1]); // second field
      fields[bs*p + 2] = discontinuous_cross_planes(coords[dim*p], coords[dim*p+1]); // third field
      /*
      fields[bs*p    ] = franke_function(coords[dim*p], coords[dim*p+1]); // first field
      fields[bs*p + 1] = franke_function(coords[dim*p], coords[dim*p+1]); // second field
      fields[bs*p + 2] = franke_function(coords[dim*p], coords[dim*p+1]); // third field
      */
    } else if (bs == 1) {
      fields[p] = franke_function(coords[dim*p], coords[dim*p+1]); // single field
      PetscPrintf(PETSC_COMM_WORLD,"point[%" PetscInt_FMT "]: %g\n", p, fields[p]);
    }
    else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported number of fields: %" PetscInt_FMT,bs);
    }
  }

  PetscCall(DMSwarmRestoreField(sw,"fields",&bs,NULL,(void **)&fields));
  PetscCall(DMSwarmRestoreField(sw,coorFieldName[0],NULL,NULL,(void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#if 0
static PetscErrorCode MatSetRHS(Partition p, PetscInt nfields, PetscReal fval[], Mat *B)
{
  Mat       RHS;
  PetscInt  f,np,npoints;
  PetscInt  *idx;
  PetscReal *LA_rhs;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetDataPointsIdx(p,&npoints,&idx));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,nfields,NULL,&RHS));
  PetscCall(MatSetOptionsPrefix(RHS,"rhs_"));
  PetscCall(MatSetFromOptions(RHS));
  PetscCall(MatDenseGetArray(RHS,&LA_rhs));
  /* Fill the matrix */
  for (np=0; np<npoints; np++) {
    for (f=0; f<nfields; f++) {
      LA_rhs[np*nfields + f] = fval[ idx[np]*nfields + f ]; // set the RHS values
    }
  }
  PetscCall(MatDenseRestoreArray(RHS,&LA_rhs));
  *B = RHS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLocalCoefficients(
  Partition p,
  PetscInt nfields,
  PetscReal fval[]
)
{
  Mat      A,B,X;
  PetscFunctionBeginUser;

  PetscCall(PartitionGetCoefficientsMat(p,&X));
  PetscCall(PartitionGetRBFMatrix(p,&A));
  PetscCall(PartitionMatSetRHS(p,nfields,fval,&B));
  /* solve the linear system A*X = B */
  PetscCall(MatMatSolve(A,B,X));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveInterpolationCoefficients(PartitionOfUnity pu, Partition *partitions, DM swarm)
{
  PetscInt  n,npart,nfields;
  PetscReal *values;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(DMSwarmGetField(swarm,"fields",&nfields,NULL,(void **)&values));
  for (n=0; n<npart; n++) {
    Partition p = partitions[n];
    PetscInt  npoints;
    PetscCall(PartitionGetDataPointsIdx(p,&npoints,NULL));
    if (npoints == 0) continue; // no data points in this partition, skip
    PetscCall(SolveLocalCoefficients(p,nfields,values));
  }
  /* Restore the field */
  PetscCall(DMSwarmRestoreField(swarm,"fields",&nfields,NULL,(void **)&values));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode PointWiseInterpolation_MatMult(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd)
{
  DMSwarmCellDM  celldm_i,celldm_d;
  DM             da;
  PetscInt       k,np,dim,npart,npoints_i,nfields,nel,nen;
  PetscInt       *cellid_i;
  const PetscInt *elidx;
  PetscReal      *coords_i,*coords_d,*weights,*values_i;
  const char     **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(PetscTime(&time[0]));

  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_i,NULL,&coordFieldName_i));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coordFieldName_i[0], NULL, NULL, (void **)&coords_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMSwarmGetField(swi, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swi, "fields",&nfields,NULL,(void **)&values_i));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd,&celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d,NULL,&coordFieldName_d));
  /* Get fields */
  PetscCall(DMSwarmGetField(swd, coordFieldName_d[0], NULL, NULL, (void **)&coords_d));

  for (np=0; np<npoints_i; np++) {
    Mat       U;
    Vec       V,W;
    PetscReal *LA_w,*LA_v;
    PetscInt  cidx = cellid_i[np]; // cell index for the particle p

    /* Create a matrix (nfields x npart) to hold the interpolated values */
    PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,nfields,npart,NULL,&U));
    
    /* Create a vector holding the interpolated fields for this point */
    //PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_values_",1,PETSC_DECIDE,nfields,&V));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,nfields,&V));

    /* Create a vector holding the weights of all partitions for this point */
    //PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_weights_",1,PETSC_DECIDE,npart,&W));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,npart,&W));
    PetscCall(VecZeroEntries(W));
    PetscCall(VecGetArray(W,&LA_w));
    
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      Vec       R,Up;
      PetscInt  nd,puidx,ndata;
      PetscInt  *data_idx;
      PetscReal *rbf;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the weights for this partition */
      LA_w[puidx] = weights[np*npart + puidx];
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      /* Get the vector of the corresponding partition in the matrix */
      PetscCall(MatDenseGetColumnVec(U,puidx,&Up));

      /* Create vector for the RBF */
      //PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_rbf_",1,PETSC_DECIDE,ndata,&R));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,ndata,&R));
      PetscCall(VecGetArray(R,&rbf));
      for (nd=0; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          dist += (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]) * (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]);
        }
        dist = PetscSqrtReal(dist);
        rbf[nd] = KernelWendlandC2(dist,p->epsilon);
      }
      PetscCall(VecRestoreArray(R,&rbf));
      /* U[:,puidx] = X^T * R */
      PetscCall(MatMultTranspose(X,R,Up));
      /* Clean-up */
      PetscCall(MatDenseRestoreColumnVec(U,puidx,&Up));
      PetscCall(VecDestroy(&R));
    }
    PetscCall(VecRestoreArray(W,&LA_w));
    /* Here compute vec(V) = mat(U) * vec(W) */
    PetscCall(MatMult(U,W,V));
    /* Set the values in the fields */
    PetscCall(VecGetArray(V,&LA_v));
    memcpy(&values_i[nfields*np],LA_v,nfields*sizeof(PetscReal));
    PetscCall(VecRestoreArray(V,&LA_v));
    /* Clean-up */
    PetscCall(MatDestroy(&U));
    PetscCall(VecDestroy(&V));
    PetscCall(VecDestroy(&W));
  }
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,"fields",NULL,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointWiseInterpolation(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd)
{
  DMSwarmCellDM  celldm_i,celldm_d;
  DM             da;
  PetscInt       k,np,dim,npart,npoints_i,nfields,nel,nen;
  PetscInt       *cellid_i;
  const PetscInt *elidx;
  PetscReal      *coords_i,*coords_d,*weights,*values_i,*interpolant;
  const char     **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;

  PetscCall(PetscTime(&time[0]));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_i,NULL,&coordFieldName_i));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coordFieldName_i[0], NULL, NULL, (void **)&coords_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMSwarmGetField(swi, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swi, "fields",&nfields,NULL,(void **)&values_i));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd,&celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d,NULL,&coordFieldName_d));
  /* Get fields */
  PetscCall(DMSwarmGetField(swd, coordFieldName_d[0], NULL, NULL, (void **)&coords_d));
  PetscCalloc1(nfields,&interpolant); // allocate memory for the interpolant

  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) {
      values_i[nfields*np + f] = 0.0; // set the initial value to zero
    }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata;
      PetscInt  *data_idx;
      PetscReal *coeffs;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));
      for (f=0; f<nfields; f++) { interpolant[f] = 0.0;}
      /* TODO: Replace the interpolation on scalar by linear algebra */
      for (nd=0; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          dist += (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]) * (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]);
        }
        dist = PetscSqrtReal(dist);
        rbf  = KernelWendlandC2(dist,p->epsilon);
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      for (f=0; f<nfields; f++) {
        /* Normalize the interpolant with the weights */
        values_i[nfields*np + f] += interpolant[f] * weights[np*npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
    }
  }
  PetscCall(PetscFree(interpolant)); // free the interpolant array
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,"fields",NULL,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionInterpolationAVX256_2D(
  Partition p,
  PetscInt  point_idx,
  PetscInt  nfields,
  PetscInt  puidx,
  PetscInt  npart,
  PetscReal *coords_i,
  PetscReal *coords_d,
  PetscReal *interpolant,
  __m256d   *interpolant_vec,
  PetscReal *values_i,
  PetscReal *weights
)
{
  Mat       X;
  PetscInt  f,nd,ndata,nd_aligned;
  PetscInt  *data_idx;
  PetscReal *coeffs;
  __m256d   epsilon,one,four,five;

  PetscFunctionBeginUser;

  PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
  if (ndata == 0) { PetscFunctionReturn(PETSC_SUCCESS); } // no data points in this partition, skip
  /* Get the coefficients of that partition */
  PetscCall(PartitionGetCoefficientsMat(p,&X));
  PetscCall(MatDenseGetArray(X,&coeffs));

  /* Set up AVX numbers */
  epsilon = _mm256_set1_pd(p->epsilon);
  one     = _mm256_set1_pd(1.0);
  four    = _mm256_set1_pd(4.0);
  five    = _mm256_set1_pd(5.0);

  /* Initialize the interpolant */
  for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm256_setzero_pd(); }
  /* Get the number of data points by chuck of 4 */
  nd_aligned = (ndata / 4) * 4;
  for (nd=0; nd<nd_aligned; nd+=4) {
    // Step 1: Compute RBF values for this chunck of 4 points
    __m256d dx,dy,dist2,dist,u,mask,u2,u4,four_u,inner,rbf;
    dx = _mm256_set_pd(
      coords_d[2*data_idx[nd+3] + 0] - coords_i[2*point_idx + 0],
      coords_d[2*data_idx[nd+2] + 0] - coords_i[2*point_idx + 0],
      coords_d[2*data_idx[nd+1] + 0] - coords_i[2*point_idx + 0],
      coords_d[2*data_idx[nd+0] + 0] - coords_i[2*point_idx + 0]
    );
    dy = _mm256_set_pd(
      coords_d[2*data_idx[nd+3] + 1] - coords_i[2*point_idx + 1],
      coords_d[2*data_idx[nd+2] + 1] - coords_i[2*point_idx + 1],
      coords_d[2*data_idx[nd+1] + 1] - coords_i[2*point_idx + 1],
      coords_d[2*data_idx[nd+0] + 1] - coords_i[2*point_idx + 1]
    );
    dist2  = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));    // dx*dx + dy*dy
    dist   = _mm256_sqrt_pd(dist2);                             // sqrt(dx*dx + dy*dy)
    u      = _mm256_sub_pd(one, _mm256_mul_pd(epsilon, dist));  // u = 1 - epsilon * dist
    mask   = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
    u2     = _mm256_mul_pd(u, u);                               // u^2
    u4     = _mm256_mul_pd(u2, u2);                             // u^4
    four_u = _mm256_mul_pd(four, u);                            // 4 * u
    inner  = _mm256_sub_pd(five, four_u);                       // 5 - 4 * u
    rbf    = _mm256_mul_pd(u4, inner);                          // rbf = u^4 * (5 - 4 * u)
    rbf    = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask);  // apply mask to zero out values where u < 0
    /* Compute the dot product R^T * X for each field */
    for (f=0; f<nfields; f++) {
      __m256d coeff_v;
      coeff_v = _mm256_loadu_pd(&coeffs[f * ndata + nd]);
      interpolant_vec[f] = _mm256_fmadd_pd(rbf, coeff_v, interpolant_vec[f]);
    }
  }
  /* Extract the chuck of 4 */
  for (f=0; f<nfields; f++) {
    __m128d hi128,lo128,sum128,final;
    hi128  = _mm256_extractf128_pd(interpolant_vec[f], 1);
    lo128  = _mm256_castpd256_pd128(interpolant_vec[f]);
    sum128 = _mm_add_pd(lo128, hi128);
    final  = _mm_hadd_pd(sum128, sum128);
    interpolant[f] = _mm_cvtsd_f64(final);
  }
  
  /* remaining data points that are not a multiple of 4 */
  for (nd=nd_aligned; nd<ndata; nd++) {
    PetscInt  d;
    PetscReal dist,rbf;
    /* Compute the RBF value for the interpolation point */
    dist = 0.0;
    for (d=0; d<2; d++) {
      dist += (coords_i[2*point_idx + d] - coords_d[2*data_idx[nd] + d]) * (coords_i[2*point_idx + d] - coords_d[2*data_idx[nd] + d]);
    }
    dist = PetscSqrtReal(dist);
    rbf  = KernelWendlandC2(dist,p->epsilon);
    for (f=0; f<nfields; f++) {
      /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
      interpolant[f] += rbf * coeffs[f*ndata + nd];
    }
  }
  /* Compute the global interpolation */
  for (f = 0; f < nfields; ++f) {
    values_i[nfields*point_idx + f] += interpolant[f] * weights[point_idx * npart + puidx];
  }
  /* Restore the coefficients array */
  PetscCall(MatDenseRestoreArray(X,&coeffs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointWiseInterpolation_AVX256_v2(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd)
{
  DMSwarmCellDM  celldm_i,celldm_d;
  DM             da;
  PetscInt       k,np,dim,npart,npoints_i,nfields,nel,nen;
  PetscInt       *cellid_i;
  const PetscInt *elidx;
  PetscReal      *coords_i,*coords_d,*weights,*values_i,*interpolant;
  const char     **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscLogDouble time[2];

  __m256d *interpolant_vec;

  PetscFunctionBeginUser;

  PetscCall(PetscTime(&time[0]));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_i,NULL,&coordFieldName_i));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coordFieldName_i[0], NULL, NULL, (void **)&coords_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMSwarmGetField(swi, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swi, "fields",&nfields,NULL,(void **)&values_i));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd,&celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d,NULL,&coordFieldName_d));
  /* Get fields */
  PetscCall(DMSwarmGetField(swd, coordFieldName_d[0], NULL, NULL, (void **)&coords_d));

  PetscCall(PetscMalloc1(nfields,&interpolant)); // allocate memory for the interpolant
  interpolant_vec = (__m256d *)_mm_malloc(nfields * sizeof(__m256d), 32); // allocate memory for the avx interpolant

  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) { values_i[nfields*np + f] = 0.0; }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata,nd_aligned;
      PetscInt  *data_idx;
      PetscReal *coeffs;
      __m256d   eps_v,one_v;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];

      PetscCall(PartitionInterpolationAVX256_2D(p,np,nfields,puidx,npart,coords_i,coords_d,interpolant,interpolant_vec,values_i,weights));

#if 0
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));

      /* Initialize the interpolant */
      for (f=0; f<nfields; f++) { interpolant_vec[f] = _mm256_setzero_pd(); }
      nd_aligned = (ndata / 4) * 4;
      eps_v = _mm256_set1_pd(p->epsilon);
      one_v = _mm256_set1_pd(1.0);
      /* NOTE: Only 2D version here */
      for (nd=0; nd<nd_aligned; nd+=4) {
        // Step 1: Compute RBF values for this chunck of 4 points
        __m256d dx,dy,dist2,dist,u,mask,u2,u4,four_u,inner,rbf;
        dx = _mm256_set_pd(
          coords_d[2*data_idx[nd+3] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+2] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+1] + 0] - coords_i[2*np + 0],
          coords_d[2*data_idx[nd+0] + 0] - coords_i[2*np + 0]
        );
        dy = _mm256_set_pd(
          coords_d[2*data_idx[nd+3] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+2] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+1] + 1] - coords_i[2*np + 1],
          coords_d[2*data_idx[nd+0] + 1] - coords_i[2*np + 1]
        );
        dist2  = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy)); // dx*dx + dy*dy
        dist   = _mm256_sqrt_pd(dist2); // sqrt(dx*dx + dy*dy)
        u      = _mm256_sub_pd(one_v, _mm256_mul_pd(eps_v, dist)); // u = 1 - epsilon * dist
        mask   = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ); // mask for u >= 0
        u2     = _mm256_mul_pd(u, u); // u^2
        u4     = _mm256_mul_pd(u2, u2); // u^4
        four_u = _mm256_mul_pd(_mm256_set1_pd(4.0), u); // 4 * u
        inner  = _mm256_sub_pd(_mm256_set1_pd(5.0), four_u); // 5 - 4 * u
        rbf    = _mm256_mul_pd(u4, inner); // rbf = u^4 * (5 - 4 * u)
        rbf    = _mm256_blendv_pd(_mm256_setzero_pd(), rbf, mask); // apply mask to zero out values where u < 0

        for (f=0; f<nfields; f++) {
          __m256d coeff_v;
          coeff_v = _mm256_loadu_pd(&coeffs[f * ndata + nd]);
          interpolant_vec[f] = _mm256_fmadd_pd(rbf, coeff_v, interpolant_vec[f]);
        }
      }

      for (f=0; f<nfields; f++) {
        __m128d hi128,lo128,sum128,final;
        hi128  = _mm256_extractf128_pd(interpolant_vec[f], 1);
        lo128  = _mm256_castpd256_pd128(interpolant_vec[f]);
        sum128 = _mm_add_pd(lo128, hi128);
        final  = _mm_hadd_pd(sum128, sum128);
        interpolant[f] = _mm_cvtsd_f64(final);
      }
      

      // remaining data points that are not a multiple of 4
      for (nd=nd_aligned; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;
        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<dim; d++) {
          dist += (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]) * (coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d]);
        }
        dist = PetscSqrtReal(dist);
        rbf  = KernelWendlandC2(dist,p->epsilon);
        for (f=0; f<nfields; f++) {
          /* multiply the RBF value by the coefficient and accumulate the RBF value for each field  */
          interpolant[f] += rbf * coeffs[f*ndata + nd];
        }
      }
      for (f = 0; f < nfields; ++f) {
        values_i[nfields*np + f] += interpolant[f] * weights[np * npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
#endif
    }
  }
  _mm_free(interpolant_vec);
  PetscCall(PetscFree(interpolant));
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,"fields",NULL,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode PointWiseInterpolation_AVX256_v1(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd)
{
  DMSwarmCellDM  celldm_i,celldm_d;
  DM             da;
  PetscInt       k,np,dim,npart,npoints_i,nfields,nel,nen;
  PetscInt       *cellid_i;
  const PetscInt *elidx;
  PetscReal      *coords_i,*coords_d,*weights,*values_i,*interpolant;
  const char     **coordFieldName_i,**coordFieldName_d,*cellFieldName_i;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;

  PetscCall(PetscTime(&time[0]));
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(PUGetNPartitions(pu,&npart));
  PetscCall(PUGetDimension(pu,&dim));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swi,&npoints_i));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swi,&celldm_i));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_i,NULL,&coordFieldName_i));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(celldm_i,&cellFieldName_i));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coordFieldName_i[0], NULL, NULL, (void **)&coords_i));
  PetscCall(DMSwarmGetField(swi, cellFieldName_i, NULL, NULL, (void **)&cellid_i));
  PetscCall(DMSwarmGetField(swi, "weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swi, "fields",&nfields,NULL,(void **)&values_i));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swd,&celldm_d));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm_d,NULL,&coordFieldName_d));
  /* Get fields */
  PetscCall(DMSwarmGetField(swd, coordFieldName_d[0], NULL, NULL, (void **)&coords_d));

  PetscCall(PetscMalloc1(nfields,&interpolant)); // allocate memory for the interpolant

  for (np=0; np<npoints_i; np++) {
    PetscInt f;
    PetscInt cidx = cellid_i[np]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    for (f=0; f<nfields; f++) { values_i[nfields*np + f] = 0.0; }
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Partition p;
      Mat       X;
      PetscInt  nd,puidx,ndata,nd_aligned;
      PetscInt  *data_idx;
      PetscReal *coeffs,*rbf_val;

      puidx = elidx[nen*cidx + k];
      p     = partitions[puidx];
      PetscCall(PartitionGetDataPointsIdx(p,&ndata,&data_idx));
      if (ndata == 0) continue; // no data points in this partition, skip
      /* Get the coefficients of that partition */
      PetscCall(PartitionGetCoefficientsMat(p,&X));
      PetscCall(MatDenseGetArray(X,&coeffs));

      // Step 1: Compute RBF values for this point
      PetscCall(PetscMalloc1(ndata,&rbf_val)); // consider allocating an array of size max_ndata outside the pointwise loop
#if 0
      // 1. Try to vectorize this loop
      // 2. Try to fuse the 2 loops
      for (nd=0; nd<ndata; nd++) {
        PetscReal dist = 0.0;
        for (d=0; d<dim; d++) {
          PetscReal delta = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += delta * delta;
        }
        rbf_val[nd] = KernelWendlandC2(PetscSqrtReal(dist), p->epsilon);
      }
#endif
      nd_aligned = (ndata / 4) * 4;
      double dist2_buf[4] __attribute__((aligned(32)));
      __m256d eps_v = _mm256_set1_pd(p->epsilon);
      __m256d one_v = _mm256_set1_pd(1.0);

      if (dim == 2) {
        for (nd = 0; nd < nd_aligned; nd += 4) {
          __m256d dx = _mm256_set_pd(
            coords_d[2*data_idx[nd+3] + 0] - coords_i[2*np + 0],
            coords_d[2*data_idx[nd+2] + 0] - coords_i[2*np + 0],
            coords_d[2*data_idx[nd+1] + 0] - coords_i[2*np + 0],
            coords_d[2*data_idx[nd+0] + 0] - coords_i[2*np + 0]
          );
          __m256d dy = _mm256_set_pd(
            coords_d[2*data_idx[nd+3] + 1] - coords_i[2*np + 1],
            coords_d[2*data_idx[nd+2] + 1] - coords_i[2*np + 1],
            coords_d[2*data_idx[nd+1] + 1] - coords_i[2*np + 1],
            coords_d[2*data_idx[nd+0] + 1] - coords_i[2*np + 1]
          );

          __m256d dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
          _mm256_store_pd(dist2_buf, dist2);

          // Now apply KernelWendlandC2
          __m256d r_v = _mm256_set_pd(
            PetscSqrtReal(dist2_buf[3]),
            PetscSqrtReal(dist2_buf[2]),
            PetscSqrtReal(dist2_buf[1]),
            PetscSqrtReal(dist2_buf[0])
          );

          __m256d u = _mm256_sub_pd(one_v, _mm256_mul_pd(eps_v, r_v));
          __m256d mask = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ);
          __m256d u2 = _mm256_mul_pd(u, u);
          __m256d u4 = _mm256_mul_pd(u2, u2);
          __m256d four_u = _mm256_mul_pd(_mm256_set1_pd(4.0), u);
          __m256d inner = _mm256_sub_pd(_mm256_set1_pd(5.0), four_u);
          __m256d phi = _mm256_mul_pd(u4, inner);
          phi = _mm256_blendv_pd(_mm256_setzero_pd(), phi, mask);
          _mm256_store_pd(&rbf_val[nd], phi); // rbf not aligned, consider aligning to replace by _mm256_store_pd()
        }
      } else if (dim == 3) {
        for (nd = 0; nd < nd_aligned; nd += 4) {
          __m256d dx = _mm256_set_pd(
            coords_d[3*data_idx[nd+3] + 0] - coords_i[3*np + 0],
            coords_d[3*data_idx[nd+2] + 0] - coords_i[3*np + 0],
            coords_d[3*data_idx[nd+1] + 0] - coords_i[3*np + 0],
            coords_d[3*data_idx[nd+0] + 0] - coords_i[3*np + 0]
          );
          __m256d dy = _mm256_set_pd(
            coords_d[3*data_idx[nd+3] + 1] - coords_i[3*np + 1],
            coords_d[3*data_idx[nd+2] + 1] - coords_i[3*np + 1],
            coords_d[3*data_idx[nd+1] + 1] - coords_i[3*np + 1],
            coords_d[3*data_idx[nd+0] + 1] - coords_i[3*np + 1]
          );
          __m256d dz = _mm256_set_pd(
            coords_d[3*data_idx[nd+3] + 2] - coords_i[3*np + 2],
            coords_d[3*data_idx[nd+2] + 2] - coords_i[3*np + 2],
            coords_d[3*data_idx[nd+1] + 2] - coords_i[3*np + 2],
            coords_d[3*data_idx[nd+0] + 2] - coords_i[3*np + 2]
          );

          __m256d dist2 = _mm256_fmadd_pd(dx, dx,
                              _mm256_fmadd_pd(dy, dy,
                                              _mm256_mul_pd(dz, dz)));
          _mm256_store_pd(dist2_buf, dist2);

          __m256d r_v = _mm256_set_pd(
            PetscSqrtReal(dist2_buf[3]),
            PetscSqrtReal(dist2_buf[2]),
            PetscSqrtReal(dist2_buf[1]),
            PetscSqrtReal(dist2_buf[0])
          );

          __m256d u = _mm256_sub_pd(one_v, _mm256_mul_pd(eps_v, r_v));
          __m256d mask = _mm256_cmp_pd(u, _mm256_setzero_pd(), _CMP_GE_OQ);
          __m256d u2 = _mm256_mul_pd(u, u);
          __m256d u4 = _mm256_mul_pd(u2, u2);
          __m256d four_u = _mm256_mul_pd(_mm256_set1_pd(4.0), u);
          __m256d inner = _mm256_sub_pd(_mm256_set1_pd(5.0), four_u);
          __m256d phi = _mm256_mul_pd(u4, inner);
          phi = _mm256_blendv_pd(_mm256_setzero_pd(), phi, mask);
          _mm256_store_pd(&rbf_val[nd], phi); // rbf not aligned, consider aligning to replace by _mm256_store_pd()
        }
      }
      // Remainder (optional)
      for (nd = nd_aligned; nd < ndata; ++nd) {
        PetscReal dist = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal delta = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += delta * delta;
        }
        PetscReal r = PetscSqrtReal(dist);
        PetscReal u = 1.0 - p->epsilon * r;
        rbf_val[nd] = (u < 0.0) ? 0.0 : PetscPowReal(u, 4) * (5.0 - 4.0 * u);
      }

      // Step 2: Vectorized loop over ndata in chunks of 4
      nd_aligned = (ndata / 4) * 4;
      for (f = 0; f < nfields; ++f) {
        __m256d accum = _mm256_setzero_pd();
        for (nd = 0; nd < nd_aligned; nd += 4) {
          __m256d rbf_v   = _mm256_load_pd(&rbf_val[nd]);                      // unaligned load
          __m256d coeff_v = _mm256_load_pd(&coeffs[f * ndata + nd]);           // aligned load
          accum = _mm256_fmadd_pd(rbf_v, coeff_v, accum);
        }
#if 0
        // Horizontal sum of accum
        double tmp[4] __attribute__((aligned(32)));
        _mm256_store_pd(tmp, accum);
        interpolant[f] = tmp[0] + tmp[1] + tmp[2] + tmp[3]; // See how to improve this
#endif
        __m128d hi128  = _mm256_extractf128_pd(accum, 1);
        __m128d lo128  = _mm256_castpd256_pd128(accum);
        __m128d sum128 = _mm_add_pd(lo128, hi128);
        __m128d final  = _mm_hadd_pd(sum128, sum128);
        interpolant[f] = _mm_cvtsd_f64(final);
        // Remainder loop
        for (nd = nd_aligned; nd < ndata; ++nd) {
          interpolant[f] += rbf_val[nd] * coeffs[f * ndata + nd];
        }
      }
      // Step 3: Normalize and accumulate
      for (f = 0; f < nfields; ++f) {
        values_i[nfields*np + f] += interpolant[f] * weights[np * npart + puidx];
      }
      /* Restore the coefficients array */
      PetscCall(MatDenseRestoreArray(X,&coeffs));
      PetscCall(PetscFree(rbf_val));
    }
  }
  PetscCall(PetscFree(interpolant));
  PetscCall(DMSwarmRestoreField(swi,coordFieldName_i[0],NULL,NULL,(void **)&coords_i));
  PetscCall(DMSwarmRestoreField(swi,cellFieldName_i,NULL,NULL,(void **)&cellid_i));
  PetscCall(DMSwarmRestoreField(swi,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swi,"fields",NULL,NULL,(void **)&values_i));
  PetscCall(DMSwarmRestoreField(swd,coordFieldName_d[0],NULL,NULL,(void **)&coords_d));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]] interpolated %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints_i,time[1]-time[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Test()
{
  LagRBFCtx        ctx;
  PartitionOfUnity pu;
  Partition        *partitions;
  DM               swarm_interp,swarm_data;
  PetscInt         k,d;
  PetscReal        overlap,radius,epsilon,box_length;
  PetscBool        exists;
  const char       output_dir[] = "output";
  PetscFunctionBeginUser;

  PetscCall(TestDirectory(output_dir,'w',&exists));
  if (!exists) { PetscCall(CreateDirectory(output_dir)); }
  
  PetscCall(LagRBFCtxCreate(&ctx));
  PetscCall(LagRBFSetFromOptions(ctx));
  PetscCall(LagRBF_PartitionOfUnityCreate(ctx));
  PetscCall(LagRBF_PartitionCreate(ctx));

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));

  /* Set epsilon parameter */
  box_length = 0.0;
  for (d=0; d<pu->dim; d++) {
    box_length = PetscMax(box_length,(pu->L[d] - pu->O[d]));
  }
  overlap = 1.5*box_length;
  radius  = overlap / PetscPowReal(pu->npart,1.0/pu->dim);//PetscSqrtReal(pu->npart);
  epsilon = 1.0 / radius;
  for (k=0; k<pu->npart; k++) { 
    PetscCall(PartitionSetEpsilon(partitions[k],epsilon)); // set the epsilon for the RBF kernel
  } 
  PetscCall(PartitionsSetIndex(pu,partitions));
  /* Set interpolation points */
  PetscCall(SwarmInterpSetUp(ctx,&swarm_interp));
  /* Set data points */
  PetscCall(CreateDataPoints(ctx,&swarm_data));
  PetscCall(SwarmDataSetValues(swarm_data));
  PetscCall(PartitionSetUp_All(partitions,pu->npart,swarm_data));

  PetscCall(PartitionSolveInterpolationCoefficients(pu,partitions,swarm_data));
  //PetscCall(PointWiseInterpolation(pu,partitions,swarm_interp,swarm_data));
  //PetscCall(PointWiseInterpolation_MatMult(pu,partitions,swarm_interp,swarm_data));
  PetscCall(RBFInterpolation(pu,partitions,swarm_interp,swarm_data));
  //PetscCall(PointWiseInterpolation_AVX256_v1(pu,partitions,swarm_interp,swarm_data));
  //PetscCall(PointWiseInterpolation_AVX256_v2(pu,partitions,swarm_interp,swarm_data));
  /* Output */
  //PetscCall(PartitionOfUnityMeshView(pu));
  /* output swarm data */
  PetscCall(SwarmOutputCoordsPetscVec(swarm_data,output_dir,NULL));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_data,output_dir,NULL));
  /* output interpolation points */
  PetscCall(SwarmOutputCoordsPetscVec(swarm_interp,output_dir,NULL));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_interp,output_dir,NULL));

  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  //PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(LagRBFInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}