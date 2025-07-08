#include <petsc.h>
#include "advection/advection_ctx.h"

PetscErrorCode AdvectionExplicit_Scalar(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscInt np,d;
  PetscLogDouble time[2];
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  for (np=0; np<npoints; np++) {
    for (d=0; d<dim; d++) {
      coords[dim*np + d] = x0[dim*np + d] + dt * velocity[d*npoints + np];
    }
  }
  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]: Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(__AVX__)
#include <immintrin.h>

PetscErrorCode AdvectionExplicit_AVX256_2D(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscInt np;
  PetscInt np_aligned = (npoints / 4) * 4;
  __m256d dt_v = _mm256_set1_pd(dt);
  PetscFunctionBeginUser;

  for (np=0; np<np_aligned; np+=4) {
    __m256d x0_x,x0_y,vx,vy,x,y,xy_lo,xy_hi,xy0,xy1;
    /* Load initial coordinates */    
    x0_x = _mm256_set_pd(x0[2*(np+3)],x0[2*(np+2)],x0[2*(np+1)],x0[2*np]);
    x0_y = _mm256_set_pd(x0[2*(np+3)+1],x0[2*(np+2)+1],x0[2*(np+1)+1],x0[2*np+1]);

    /* Load velocities */
#if (PETSC_MEMALIGN == 32)
    vx = _mm256_load_pd(&velocity[0*npoints + np]);
    vy = _mm256_load_pd(&velocity[1*npoints + np]);
#else
    vx = _mm256_loadu_pd(&velocity[0*npoints + np]);
    vy = _mm256_loadu_pd(&velocity[1*npoints + np]);
#endif
    /* Compute new coordinates */
    x = _mm256_fmadd_pd(dt_v, vx, x0_x); // x0 + dt * vx
    y = _mm256_fmadd_pd(dt_v, vy, x0_y); // y0 + dt * vy
    /* Store new coordinates */
    xy_lo = _mm256_unpacklo_pd(x, y); // [x0 y0 x2 y2]
    xy_hi = _mm256_unpackhi_pd(x, y); // [x1 y1 x3 y3]

    // Shuffle to get [x0 y0 x1 y1] and [x2 y2 x3 y3]
    xy0 = _mm256_permute2f128_pd(xy_lo, xy_hi, 0x20); // lower 128s: [x0 y0 x1 y1]
    xy1 = _mm256_permute2f128_pd(xy_lo, xy_hi, 0x31); // upper 128s: [x2 y2 x3 y3]

    // Store to coords array
#if (PETSC_MEMALIGN == 32)
    _mm256_store_pd(&coords[2*np + 0], xy0);
    _mm256_store_pd(&coords[2*np + 4], xy1);
#else
    _mm256_storeu_pd(&coords[2*np + 0], xy0);
    _mm256_storeu_pd(&coords[2*np + 4], xy1);
#endif
  }
  // Scalar remainder
  for (; np<npoints; np++) {
    PetscInt d;
    for (d=0; d<dim; d++) {
      coords[2*np + d] = x0[2*np + d] + dt * velocity[d*npoints + np];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionExplicit_AVX256_3D(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscInt np;
  PetscInt np_aligned = (npoints / 4) * 4;
  __m256d dt_v = _mm256_set1_pd(dt);
  PetscFunctionBeginUser;

  for (np = 0; np < np_aligned; np += 4) {
    PetscReal x_vals[4],y_vals[4],z_vals[4];
    __m256d   x0_x,x0_y,x0_z,vx,vy,vz,x,y,z;

    // Load x0 positions (AoS layout)
    x0_x = _mm256_set_pd(x0[3*(np+3)], x0[3*(np+2)], x0[3*(np+1)], x0[3*np]);
    x0_y = _mm256_set_pd(x0[3*(np+3)+1], x0[3*(np+2)+1], x0[3*(np+1)+1], x0[3*np+1]);
    x0_z = _mm256_set_pd(x0[3*(np+3)+2], x0[3*(np+2)+2], x0[3*(np+1)+2], x0[3*np+2]);

    // Load velocities from SoA layout
#if (PETSC_MEMALIGN == 32)
    vx = _mm256_load_pd(&velocity[0*npoints + np]);
    vy = _mm256_load_pd(&velocity[1*npoints + np]);
    vz = _mm256_load_pd(&velocity[2*npoints + np]);
#else
    vx = _mm256_loadu_pd(&velocity[0*npoints + np]);
    vy = _mm256_loadu_pd(&velocity[1*npoints + np]);
    vz = _mm256_loadu_pd(&velocity[2*npoints + np]);
#endif

    // Compute new positions
    x = _mm256_fmadd_pd(dt_v, vx, x0_x);
    y = _mm256_fmadd_pd(dt_v, vy, x0_y);
    z = _mm256_fmadd_pd(dt_v, vz, x0_z);

    // Extract values and store to coords[]
#if (PETSC_MEMALIGN == 32)
      _mm256_store_pd(x_vals, x);
      _mm256_store_pd(y_vals, y);
      _mm256_store_pd(z_vals, z);
#else
      /* Use unaligned store */
      _mm256_storeu_pd(x_vals, x);
      _mm256_storeu_pd(y_vals, y);
      _mm256_storeu_pd(z_vals, z);
#endif

    for (int i = 0; i < 4; ++i) {
      PetscInt idx = np + i;
      coords[3*idx + 0] = x_vals[i];
      coords[3*idx + 1] = y_vals[i];
      coords[3*idx + 2] = z_vals[i];
    }
  }

  // Scalar remainder
  for (; np < npoints; ++np) {
    for (PetscInt d = 0; d < dim; ++d) {
      coords[3*np + d] = x0[3*np + d] + dt * velocity[d*npoints + np];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionExplicit_AVX256(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscLogDouble time[2];
  PetscFunctionBeginUser;
  PetscCall(PetscTime(&time[0]));
  switch (dim) {
    case 2:
      PetscCall(AdvectionExplicit_AVX256_2D(coords, x0, velocity, dt, npoints, dim));
      break;
    case 3:
      PetscCall(AdvectionExplicit_AVX256_3D(coords, x0, velocity, dt, npoints, dim));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"AdvectionExplicit_AVX256 only supports 2D and 3D, found dim = %" PetscInt_FMT,dim);
  }
  PetscCall(PetscTime(&time[1]));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]: Advected %" PetscInt_FMT " points in %g seconds\n",PETSC_FUNCTION_NAME,npoints,time[1]-time[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode AdvectionExplicit(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscFunctionBeginUser;
#if defined(__AVX__)
  PetscCall(AdvectionExplicit_AVX256(coords, x0, velocity, dt, npoints, dim));
#else
  PetscCall(AdvectionExplicit_Scalar(coords, x0, velocity, dt, npoints, dim));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionSetInitialValues(AdvectionCtx ctx, DM swi)
{
  DMSwarmCellDM  celldm;
  PetscInt       dim,npoints;
  PetscReal      *coords,*velocity,*x0,*v_i;
  const char     *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  /* 1st order advection does not require to store initial values */
  if (ctx->scheme == RK1) { PetscFunctionReturn(PETSC_SUCCESS); }

  PetscCall(DMGetDimension(swi,&dim));
  PetscCall(DMSwarmGetLocalSize(swi,&npoints));
  PetscCall(DMSwarmGetCellDMActive(swi, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(ctx, &velocity_name));
  /* Get fields */
  PetscCall(DMSwarmGetField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  /* Store the initial coords */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* Store the velocity at initial coords: v_0 = v(x0) */
  PetscCall(PetscArraycpy(v_i,velocity,npoints*dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, "v_adv",          NULL, NULL, (void **)&v_i));
  PetscFunctionReturn(PETSC_SUCCESS);
}