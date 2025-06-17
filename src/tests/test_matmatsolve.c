#include <petsc.h>
#include "lagrangian_rbf.h"
#include "kernels_rbf.h"
#include "partition_solve.h"
#include "io_utils.h"
#include "swarm_utils.h"
#include "swarm_interp.h"
#include "swarm_data.h"

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
    PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_values_",1,PETSC_DECIDE,nfields,&V));

    /* Create a vector holding the weights of all partitions for this point */
    PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_weights_",1,PETSC_DECIDE,npart,&W));
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
      PetscCall(VecCreateFromOptions(PETSC_COMM_SELF,"interp_rbf_",1,PETSC_DECIDE,ndata,&R));
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

static PetscErrorCode PointWiseInterpolation_AVX256(PartitionOfUnity pu, Partition *partitions, DM swi, DM swd)
{
  DMSwarmCellDM  celldm_i,celldm_d;
  DM             da;
  PetscInt       d,k,np,dim,npart,npoints_i,nfields,nel,nen;
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
      for (nd=0; nd<ndata; nd++) {
        PetscReal dist = 0.0;
        for (d=0; d<dim; d++) {
          PetscReal delta = coords_i[dim*np + d] - coords_d[dim*data_idx[nd] + d];
          dist += delta * delta;
        }
        rbf_val[nd] = KernelWendlandC2(PetscSqrtReal(dist), p->epsilon);
      }

      // Step 2: Vectorized loop over ndata in chunks of 4
      nd_aligned = (ndata / 4) * 4;
      for (f = 0; f < nfields; ++f) {
        __m256d accum = _mm256_setzero_pd();
        for (nd = 0; nd < nd_aligned; nd += 4) {
          __m256d rbf_v   = _mm256_loadu_pd(&rbf_val[nd]);                      // unaligned load
          __m256d coeff_v = _mm256_load_pd(&coeffs[f * ndata + nd]);           // aligned load
          accum = _mm256_fmadd_pd(rbf_v, coeff_v, accum);
        }
        // Horizontal sum of accum
        double tmp[4] __attribute__((aligned(32)));
        _mm256_store_pd(tmp, accum);
        interpolant[f] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
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
  PetscCall(PointWiseInterpolation(pu,partitions,swarm_interp,swarm_data));
  //PetscCall(PointWiseInterpolation_MatMult(pu,partitions,swarm_interp,swarm_data));
  PetscCall(PointWiseInterpolation_AVX256(pu,partitions,swarm_interp,swarm_data));
  /* Output */
  //PetscCall(PartitionOfUnityMeshView(pu));
  /* output swarm data */
  PetscCall(SwarmOutputCoordsPetscVec(swarm_data,output_dir));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_data,output_dir));
  /* output interpolation points */
  PetscCall(SwarmOutputCoordsPetscVec(swarm_interp,output_dir));
  PetscCall(SwarmOutputFieldsPetscVec(swarm_interp,output_dir));

  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(LagRBFCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscCall(Test());

  PetscCall(PetscFinalize());
  
  return 0;
}