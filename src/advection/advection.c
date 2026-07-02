#include <petsc.h>
#include "lagrangian_rbf.h"
#include "advection/advection_ctx.h"
#include "swarm/swarm_utils.h"
#include "interpolation.h"

PetscLogEvent LAGRBF_AdvectionExplicit_scalar;

PetscErrorCode AdvectionExplicit_Scalar(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt npoints, 
  PetscInt dim
)
{
  PetscInt       np,d;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_AdvectionExplicit_scalar,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  if (coords == x0) { SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "coords and x0 point to same memory"); }
  for (np=0; np<npoints; np++) {
    for (d=0; d<dim; d++) {
      coords[dim*np + d] = x0[dim*np + d] + dt * velocity[dim*np + d];
    }
  }
  PetscCall(PetscLogEventEnd(LAGRBF_AdvectionExplicit_scalar,0,0,0,0));
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
    x0_x = _mm256_set_pd(x0[2*(np+3)  ],x0[2*(np+2)  ],x0[2*(np+1)  ],x0[2*np  ]);
    x0_y = _mm256_set_pd(x0[2*(np+3)+1],x0[2*(np+2)+1],x0[2*(np+1)+1],x0[2*np+1]);

    /* Load velocities */
    vx = _mm256_set_pd(velocity[2*(np+3)  ], velocity[2*(np+2)  ], velocity[2*(np+1)  ], velocity[2*np  ]);
    vy = _mm256_set_pd(velocity[2*(np+3)+1], velocity[2*(np+2)+1], velocity[2*(np+1)+1], velocity[2*np+1]);
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
      coords[2*np + d] = x0[2*np + d] + dt * velocity[2*np + d];
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
    __m256d   x0_x,x0_y,x0_z,vx,vy,vz,x,y,z,xy_lo,xy_hi,xz_lo,yz_hi,xyz0,xyz1,xyz2;

    // Load x0 positions (AoS layout)
    x0_x = _mm256_set_pd(x0[3*(np+3)  ], x0[3*(np+2)  ], x0[3*(np+1)  ], x0[3*np  ]);
    x0_y = _mm256_set_pd(x0[3*(np+3)+1], x0[3*(np+2)+1], x0[3*(np+1)+1], x0[3*np+1]);
    x0_z = _mm256_set_pd(x0[3*(np+3)+2], x0[3*(np+2)+2], x0[3*(np+1)+2], x0[3*np+2]);

    // Load velocities (AoS layout)
    vx = _mm256_set_pd(velocity[3*(np+3)  ], velocity[3*(np+2)  ], velocity[3*(np+1)  ], velocity[3*np  ]);
    vy = _mm256_set_pd(velocity[3*(np+3)+1], velocity[3*(np+2)+1], velocity[3*(np+1)+1], velocity[3*np+1]);
    vz = _mm256_set_pd(velocity[3*(np+3)+2], velocity[3*(np+2)+2], velocity[3*(np+1)+2], velocity[3*np+2]);

    // Compute new positions
    x = _mm256_fmadd_pd(dt_v, vx, x0_x);
    y = _mm256_fmadd_pd(dt_v, vy, x0_y);
    z = _mm256_fmadd_pd(dt_v, vz, x0_z);

    // Interleave x,y,z coordinates for 4 points in AoS format
    // We need to create: [x0 y0 z0 x1] [y1 z1 x2 y2] [z2 x3 y3 z3]
    
    // First, create intermediate vectors with proper ordering
    xy_lo = _mm256_unpacklo_pd(x, y);  // [x0 y0 x2 y2]
    xy_hi = _mm256_unpackhi_pd(x, y);  // [x1 y1 x3 y3]
    xz_lo = _mm256_unpacklo_pd(x, z);  // [x0 z0 x2 z2]
    yz_hi = _mm256_unpackhi_pd(y, z);  // [y1 z1 y3 z3]

    // Create the three output vectors for 12 coordinates (3*4 points)
    // xyz0: [x0 y0 z0 x1] - first point complete + x of second point
    xyz0 = _mm256_blend_pd(
             _mm256_blend_pd(xy_lo, z, 0x4),  // [x0 y0 z0 x2] -> blend z0 into position 2
             _mm256_permute2f128_pd(xy_hi, xy_hi, 0x00), // get x1 from xy_hi
             0x8); // blend into position 3

    // xyz1: [y1 z1 x2 y2] - rest of second point + third point without z
    xyz1 = _mm256_blend_pd(
             _mm256_permute2f128_pd(yz_hi, xz_lo, 0x21), // [y1 z1 x2 z2]
             _mm256_permute2f128_pd(xy_lo, xy_lo, 0x11),  // get y2
             0x8); // blend y2 into position 3

    // xyz2: [z2 x3 y3 z3] - rest of third point + fourth point complete
    xyz2 = _mm256_blend_pd(
             _mm256_permute2f128_pd(z, xy_hi, 0x21),     // [z2 ? x3 y3]
             _mm256_permute2f128_pd(z, z, 0x11),         // get z3
             0x8); // blend z3 into position 3

    // Store the interleaved coordinates
#if (PETSC_MEMALIGN == 32)
    _mm256_store_pd(&coords[3*np + 0], xyz0);  // [x0 y0 z0 x1]
    _mm256_store_pd(&coords[3*np + 4], xyz1);  // [y1 z1 x2 y2]
    _mm256_store_pd(&coords[3*np + 8], xyz2);  // [z2 x3 y3 z3]
#else
    _mm256_storeu_pd(&coords[3*np + 0], xyz0);
    _mm256_storeu_pd(&coords[3*np + 4], xyz1);
    _mm256_storeu_pd(&coords[3*np + 8], xyz2);
#endif
  }

  // Scalar remainder
  for (; np < npoints; ++np) {
    for (PetscInt d = 0; d < dim; ++d) {
      coords[3*np + d] = x0[3*np + d] + dt * velocity[3*np + d];
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
  long int       npoints_local,npoints_global;
  double         time_local,time_max;
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
  npoints_local = (long int)npoints;
  PetscCallMPI(MPI_Allreduce(&npoints_local,&npoints_global,1,MPI_LONG,MPI_SUM,PETSC_COMM_WORLD));
  time_local = (double)time[1]-(double)time[0];
  PetscCallMPI(MPI_Allreduce(&time_local,&time_max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]: Advected %ld points in %1.4e seconds\n",PETSC_FUNCTION_NAME,npoints_global,time_max);
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
#if 0 //defined(__AVX__)
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
  PetscReal      *coords,*velocity,*x0,*v0;
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
  PetscCall(DMSwarmGetField(swi, "v_adv_0",        NULL, NULL, (void **)&v0));
  /* Store the initial coords */
  PetscCall(PetscArraycpy(x0,coords,npoints*dim));
  /* Store the velocity at initial coords: v0 = v(x0) */
  PetscCall(PetscArraycpy(v0,velocity,npoints*dim));
  /* Restore fields */
  PetscCall(DMSwarmRestoreField(swi, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swi, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(swi, "coor_init",      NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(swi, "v_adv_0",        NULL, NULL, (void **)&v0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionSetVelocityFunction(AdvectionCtx ctx, PetscErrorCode (*velocity_fn)(PetscReal*,PetscReal*,void*), void *user)
{
  PetscFunctionBeginUser;
  ctx->velocity_type = ANALYTICAL;
  ctx->velocity_fn   = velocity_fn;
  ctx->user_fn       = user;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionSetVelocityFunctionBatch(AdvectionCtx ctx, PetscErrorCode (*velocity_fn_batch)(PetscInt,PetscInt,PetscReal*,PetscReal*,void*), void *user)
{
  PetscFunctionBeginUser;
  ctx->velocity_type     = ANALYTICAL;
  ctx->velocity_fn_batch = velocity_fn_batch;
  ctx->user_fn           = user;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionCopyVelocityField(LagRBFCtx ctx, DM swarm)
{
  AdvectionCtx adv_ctx;
  PetscInt     dim,npoints;
  PetscReal    *interp_fields,*velocity;
  const char   *velocity_name;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMSwarmGetField(swarm, "fields", NULL, NULL, (void **)&interp_fields));
  PetscCall(DMSwarmGetField(swarm, velocity_name, NULL, NULL, (void **)&velocity));
  /* Copy the first dim fields into velocity field */
  PetscCall(CopyContiguousBlockData(velocity, interp_fields, npoints, dim, ctx->n_ifields, 0));
  PetscCall(DMSwarmRestoreField(swarm, "fields", NULL, NULL, (void **)&interp_fields));
  PetscCall(DMSwarmRestoreField(swarm, velocity_name, NULL, NULL, (void **)&velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionEvaluateVelocityFunction(AdvectionCtx adv_ctx, DM swarm)
{
  DMSwarmCellDM celldm;
  PetscInt      np,dim,npoints;
  PetscReal     *coords,*velocity;
  const char    *velocity_name,**coorFieldName;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coorFieldName));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  PetscCall(DMSwarmGetField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(swarm, velocity_name,    NULL, NULL, (void **)&velocity));
  /* Evaluate the velocity at the current particle positions */
  if (adv_ctx->velocity_fn_batch) { /* If a batch function is available */
    PetscCall(adv_ctx->velocity_fn_batch(npoints, dim, coords, velocity, adv_ctx->user_fn));
  } else { /* Point-wise evaluation */
    PetscCheck(adv_ctx->velocity_fn,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"AdvectionCtx->velocity_fn is NULL, please provide a valid function");
    for (np=0; np<npoints; np++) {
      PetscCall(adv_ctx->velocity_fn(&coords[dim*np], &velocity[dim*np], adv_ctx->user_fn));
    }
  }
  #if 0
  PetscCheck(adv_ctx->velocity_fn,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"AdvectionCtx->velocity_fn is NULL, please provide a valid function");
  for (np=0; np<npoints; np++) {
    adv_ctx->velocity_fn(&coords[dim*np], &velocity[dim*np], adv_ctx->user_fn);
  }
  #endif
  PetscCall(DMSwarmRestoreField(swarm, coorFieldName[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(swarm, velocity_name,    NULL, NULL, (void **)&velocity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionInitializeTimestepVelocity(LagRBFCtx ctx, DM swarm)
{
  AdvectionCtx adv_ctx;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  LagRBFGetAdvectionCtx(ctx,&adv_ctx);
  switch (adv_ctx->velocity_type) {
    case INTERPOLATION:
      PetscCall(AdvectionCopyVelocityField(ctx,swarm));
      break;

    case ANALYTICAL:
      PetscCall(AdvectionEvaluateVelocityFunction(adv_ctx,swarm));
      break;
    
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"AdvectionCtx->velocity_type has an invalid value: %d",adv_ctx->velocity_type);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AdvectionEvaluateVelocity - Evaluate the velocity field at the current particle positions.

  Input Parameters:
+ ctx  - The `LagRBFCtx` context
. swi  - The `DM` swarm containing the interpolation points (particles)
- swd  - The `DM` swarm containing the data points (velocity field). Can be NULL if using an analytical velocity field.
  Notes:
  This function uses the velocity type specified in the `AdvectionCtx` to either:
    1) Use RBF interpolation to evaluate the velocity at the particle positions
    2) Use a user-defined function to evaluate the velocity at the particle positions
@*/
PetscErrorCode AdvectionEvaluateVelocity(LagRBFCtx ctx, DM swi, DM swd)
{
  AdvectionCtx adv_ctx;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  switch (adv_ctx->velocity_type) {
    case INTERPOLATION:
    {
      const char *velocity_name;
      PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
      PetscCall(RBFInterpolation(ctx,swi,swd,velocity_name));
    }
      break;

    case ANALYTICAL:
      PetscCall(AdvectionEvaluateVelocityFunction(adv_ctx,swi));
      break;
    
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"AdvectionCtx->velocity_type has an invalid value: %d",adv_ctx->velocity_type);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionComputeTimestep_CFL(LagRBFCtx ctx, DM swarm, PetscReal *_dt)
{
  PartitionOfUnity pu;
  AdvectionCtx     adv_ctx;
  PetscInt         np,npoints,bs;
  PetscReal        epsilon,radius,constant,dt;
  PetscReal        *velocity;
  const char       *velocity_name;
  PetscFunctionBeginUser;

  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));

  PetscCall(PUGetSupportFactor(pu,&epsilon));
  radius = 1.0 / epsilon;
  constant = ctx->cfl_fac; // CFL constant

  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  PetscCall(DMSwarmGetField(swarm, velocity_name, &bs, NULL, (void **)&velocity));
  dt = PETSC_MAX_REAL;
  for (np=0; np<npoints; np++) {
    PetscInt  d;
    PetscReal norm_velocity,tol,dt_local;

    tol = 1e-16;
    norm_velocity = 0.0;
    for (d=0; d<bs; d++) {
      norm_velocity += velocity[bs*np + d] * velocity[bs*np + d];
    }
    norm_velocity = PetscSqrtReal(norm_velocity);
    dt_local = constant * radius / (norm_velocity + tol);
    dt = PetscMin(dt,dt_local);
  }
  PetscCall(DMSwarmRestoreField(swarm, velocity_name, &bs, NULL, (void **)&velocity));
  /* Perform reduction to get the global min */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)swarm)));
  PetscPrintf(PETSC_COMM_WORLD,"CFL timestep: dt = %g\n",dt);
  if (_dt) *_dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionProjectOntoSphereBoundary_Pointwise(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax)
{
  PetscInt  d;
  PetscReal pos;
  PetscReal normal[3];
  PetscFunctionBeginUser;
  pos = 0.0;
  for (d=0; d<dim; d++) { pos += coords[d]*coords[d]; }
  if (pos < Rmax*Rmax && pos > Rmin*Rmin ) { PetscFunctionReturn(PETSC_SUCCESS); } 
  pos = PetscSqrtReal(pos);
  for (d=0; d<dim; d++) { normal[d] = coords[d] / pos; }
  if (pos > Rmax) {
    /* Project onto outer sphere */
    for (d=0; d<dim; d++) { coords[d] = Rmax * normal[d]; }
  } else {
    /* Project onto inner sphere */
    for (d=0; d<dim; d++) { coords[d] = Rmin * normal[d]; }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionProjectOntoSphereBoundary(PetscReal *coords, PetscInt  dim, PetscReal Rmin, PetscReal Rmax, PetscInt npoints)
{
  PetscInt np;
  PetscFunctionBeginUser;
  for (np=0; np<npoints; np++) {
    PetscCall(AdvectionProjectOntoSphereBoundary_Pointwise(&coords[dim*np],dim,Rmin,Rmax));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Call after advection is done but before migrate */
PetscErrorCode AdvectionCorrectSphereBoundaryPosition_Pointwise(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt  dim,
  PetscReal Rmin2,
  PetscReal Rmax2
)
{
  PetscInt  d;
  PetscReal R2,fac;
  PetscFunctionBeginUser;

  R2 = 0.0;
  for (d=0; d<dim; d++) { R2 += coords[d]*coords[d]; }
  if (R2 > Rmax2 || R2 < Rmin2) {
    PetscReal a,b,c,r,delta;
    /* Solve a quadratic equation for the intersection with the sphere */
    if (R2 > Rmax2) { r = Rmax2; } 
    else            { r = Rmin2; }
    a = 0.0; b = 0.0; c = -r;
    for (d=0; d<dim; d++) {
      a += velocity[d]*velocity[d];
      b += 2.0 * velocity[d]*x0[d];
      c += x0[d]*x0[d];
    }
    delta = b*b - 4.0*a*c;
    if (delta < 0.0) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Negative discriminant in sphere intersection computation: delta = %g",delta);
    }
    delta = PetscSqrtReal(delta);
    /* We only keep the positive root */
    fac = (-b + delta) / (2.0*a);
    if (dt < 0.0) fac = -fac;  /* Reverse direction if going backwards in time */
    /* Update position */
    for (d=0; d<dim; d++) {
      coords[d] = x0[d] + fac * velocity[d];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AdvectionCorrectSphereBoundaryPosition(
  PetscReal *coords, 
  PetscReal *x0, 
  PetscReal *velocity, 
  PetscReal dt, 
  PetscInt  npoints, 
  PetscInt  dim,
  PetscReal Rmin,
  PetscReal Rmax
)
{
  PetscInt       np;
  PetscReal      Rmin2,Rmax2;
  PetscFunctionBeginUser;
  Rmin2 = Rmin*Rmin;
  Rmax2 = Rmax*Rmax;
  for (np=0; np<npoints; np++) {
    PetscCall(AdvectionCorrectSphereBoundaryPosition_Pointwise(&coords[dim*np],&x0[dim*np],&velocity[dim*np],dt,dim,Rmin2,Rmax2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}