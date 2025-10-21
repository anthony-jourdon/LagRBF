#include <petsc.h>
#include "lagrangian_rbf.h"
#include "partition.h"
#include "advection/advection_ctx.h"
#include "kernels/kernels_rbf.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"

PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX256;
PetscLogEvent LAGRBF_SwarmInterpSetWeights_Scalar;

PetscErrorCode SwarmInterpCreateAdvectionFields(LagRBFCtx ctx, DM swarm)
{
  AdvectionCtx adv_ctx;
  PetscInt     dim,order;
  const char   *velocity_name;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ Advection activated ]]: creating advection fields\n");
  
  PetscCall(DMGetDimension(swarm,&dim));
  PetscCall(LagRBFGetAdvectionCtx(ctx,&adv_ctx));
  PetscCall(AdvectionGetIntegrationOrder(adv_ctx,&order));
  PetscCall(AdvectionGetVelocityName(adv_ctx,&velocity_name));
  /* Velocity field interpolated */
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering velocity field \"%s\" of block size: %" PetscInt_FMT "\n",velocity_name,dim);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,velocity_name,dim,PETSC_REAL));
  /* If the scheme is higher than first order, we need to store the initial coordinates and the velocity at each integration step */
  if (adv_ctx->scheme != RK1 || adv_ctx->scheme != RK2) {
    PetscInt b;
    /* Store initial coordinates before advection */
    PetscPrintf(PETSC_COMM_WORLD,"\tRegistering initial coordinates field \"coor_init\" of block size: %" PetscInt_FMT "\n",dim);
    PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"coor_init",dim,PETSC_REAL));
    for (b=0; b<order; b++) {
      char v_adv_name[PETSC_MAX_PATH_LEN];
      /* Register Velocity at integration step */
      PetscCall(PetscSNPrintf(v_adv_name,PETSC_MAX_PATH_LEN-1,"v_adv_%d",b));
      PetscPrintf(PETSC_COMM_WORLD,"\tRegistering integration step velocity field \"%s\" of block size: %" PetscInt_FMT "\n",v_adv_name,dim);
      PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,v_adv_name,dim,PETSC_REAL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpCreate(LagRBFCtx ctx, DM *sw)
{
  PartitionOfUnity pu;
  DM               dap,swarm;
  PetscInt         dim,nnodes;
  char             dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  /* Create interpolation points */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(PUGetDMDA(pu,&dap));
  PetscCall(DMGetDimension(dap,&dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dap), &swarm)); // COMM_WORLD
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC)); // This sets automatically the migrate type: DMSWARM_MIGRATE_DMCELLNSCATTER
  PetscCall(DMSwarmSetCellDM(swarm, dap));
  PetscCall(PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp"));
  PetscCall(PetscObjectSetName((PetscObject)swarm,dm_name));
  PetscCall(PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name));
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  /* Weights for interpolation */
  PetscCall(PUGetNPartitionsGlobal(pu,&nnodes));
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering \"weights\" real field of block size: %" PetscInt_FMT "\n",nnodes);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"weights",nnodes,PETSC_REAL));
  /* Fields to interpolate, also contain the velocity even in advection mode */
  PetscPrintf(PETSC_COMM_WORLD,"\tRegistering \"fields\" real field of block size: %" PetscInt_FMT "\n",ctx->n_ifields);
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"fields",ctx->n_ifields,PETSC_REAL));
  /* If advection is required, special fields are added */
  if (ctx->advection) {
    PetscCall(SwarmInterpCreateAdvectionFields(ctx, swarm));
  }
  // May need to add another set of fields that need to be advected but not interpolated
  // PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"pfields",ctx->n_pfields,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSetUp(swarm));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetCoordinatesFromOptions(PartitionOfUnity pu, DM sw, InsertMode mode)
{
  DM         da=NULL;
  PetscInt   m[]={0,0,1};
  PetscInt   d;
  PetscReal  O[]={0.0, 0.0, 0.0};
  PetscReal  L[]={1.0, 1.0, 1.0};
  PetscBool  found,spherical=PETSC_FALSE;
  const char *prefix;
  PetscFunctionBeginUser;
  
  PetscCall(PUGetDMDA(pu,&da));
  PetscCall(DMGetOptionsPrefix(sw,&prefix));
  d = pu->dim;
  PetscCall(PetscOptionsGetIntArray(NULL,prefix,"-npi",m,&d,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Option -%snpi not found. User must provide the number of interpolation points in each dimension.",prefix);
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%snpi must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-O",O,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sO must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetRealArray(NULL,prefix,"-L",L,&d,&found));
  PetscCheck(d == pu->dim,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Option -%sL must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "",prefix,pu->dim,d);
  PetscCall(PetscOptionsGetBool(NULL,prefix,"-spherical",&spherical,&found));
  if (spherical) {
    PetscCall(DMSwarmSetPointsSphericalCoordinates(sw,O,L,m,mode,PETSC_TRUE));
  } else {
    PetscCall(DMSwarmSetPointsUniformCoordinates(sw,O,L,m,mode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpLoadCoordinates(PartitionOfUnity pu, DM sw)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  To compute the weights in theory we would need to compute at each point:
    w_j = phi(||x - c_j||) / (sum_{k=1}^{N} phi(||x - c_k||)) 
  where:
    - w_j is the weight for partition j
    - phi is the RBF kernel function
    - x is the coordinate of the point
    - c_j is the center of partition j
    - N is the total number of partitions
  But in practice, because the kernel has compact support i.e., 
  phi(r) = 0 for r > R (R is the support radius),
  we only need to compute the weights for the partitions 
  whose centers are within the support radius of the point.
  Except if one discretizes the domain agressively, we should only need the nodes in the local DMDA
  and thus no communication should be needed.
*/
static PetscErrorCode SwarmInterpSetWeights_Scalar(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DMSwarmCellDM    celldm;
  DM               das,cdas;
  Vec              coords;
  PetscReal        epsilon;
  PetscReal        *pcoords,*weights,*LA_coords;
  PetscInt         dim,npart,npg,npoints,nn,np,n,puidx;
  const char      **coordFields;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_Scalar,0,0,0,0));

  /* Get data structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Get the rank-local DMDA (overlapping between processes) */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  /* Get partitions centre coordinates (DMDA's nodes) */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  PetscCall(DMGetCoordinates(das,&coords));
  PetscCall(VecGetArray(coords,&LA_coords));
  /* Get the number of nodes in the local DMDA */
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Get the number of global partitions */
  PetscCall(PUGetNPartitionsGlobal(pu,&npg));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCheck(nn == npg, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of weigths must equal the number of partitions in the global mesh: %" PetscInt_FMT ", found: %" PetscInt_FMT "",npg,nn);
  PetscCall(PUGetEpsilon(pu,&epsilon));
  for (np=0; np<npoints; np++) {
    PetscReal sum_w = 0.0;
    /* Loop over partitions */
    for (n=0; n<npart; n++) {
      Partition p = partitions[n]; // current partition
      PetscInt  d;
      PetscReal dist;
      PetscReal *pu_centre;
      
      PetscCall(PartitionGetIndex(p,&puidx));
      pu_centre = &LA_coords[dim*puidx];
      dist = 0.0;
      for (d=0; d<dim; d++) { 
        dist += (pcoords[dim*np + d] - pu_centre[d]) * (pcoords[dim*np + d] - pu_centre[d]); 
      }
      dist = PetscSqrtReal(dist);
      /* Compute the weight */
      weights[np*npart + puidx] = KernelWendlandC2(dist,epsilon);
      sum_w += weights[np*npart + puidx];
    }
    /* Normalize weights */
    for (n=0; n<npart; n++) {
      Partition p = partitions[n]; // current partition

      PetscCall(PartitionGetIndex(p,&puidx));
      if (sum_w > 0.0) {
        /* Normalize the weights */
        weights[np*npart + puidx] /= sum_w;
      } else {
        /* If the sum of weights is zero, set the weight to zero */
        weights[np*npart + puidx] = 0.0;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coords,&LA_coords));

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_Scalar,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(__AVX__)
static PetscErrorCode SwarmInterpSetWeights_AVX256_2D(
  PetscInt  npart,       // number of partitions
  Partition *partitions, // array of partitions
  PetscReal *LA_coords,  // partition centres coordinates
  PetscInt  npoints,     // number of interpolation points
  PetscReal *pcoords,    // interpolation points coordinates
  PetscReal *weights,    // output weights (size npoints*npart)
  PetscReal *sum_w       // output sum of weights (size npoints)
)
{
  PetscInt d,n,np,i,np_aligned;
  PetscInt dim = 2;
  PetscFunctionBeginUser;

  np_aligned = (npoints/4)*4;
  for (n=0; n<npart; n++) {
    Partition p = partitions[n]; // current partition
    PetscInt  puidx;
    __m256d   epsilon,centre_x,centre_y;

    PetscCall(PartitionGetIndex(p,&puidx));
    epsilon   = _mm256_set1_pd(p->epsilon);
    centre_x  = _mm256_set1_pd(LA_coords[2*puidx    ]);
    centre_y  = _mm256_set1_pd(LA_coords[2*puidx + 1]);
  
    for (np=0; np<np_aligned; np+=4) {
      double  w_arr[4] __attribute__((aligned(32)));
      __m256d x,y,dx,dy,dist2,dist,rbf,sum_v;
      /* Store the coordinates */
      x = _mm256_set_pd(
        pcoords[ 2*(np+3) ],
        pcoords[ 2*(np+2) ],
        pcoords[ 2*(np+1) ],
        pcoords[ 2*np     ]
      );
      y = _mm256_set_pd(
        pcoords[ 2*(np+3) + 1 ],
        pcoords[ 2*(np+2) + 1 ],
        pcoords[ 2*(np+1) + 1 ],
        pcoords[ 2*np     + 1 ]
      );
      /* Compute the distance between the point and the partition centre */
      dx    = _mm256_sub_pd(x, centre_x); // dx = x - centre_x
      dy    = _mm256_sub_pd(y, centre_y); // dy = y - centre_y
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy)); // dx*dx + dy*dy
      dist  = _mm256_sqrt_pd(dist2);                          // sqrt(dx*dx + dy*dy)
      rbf   = KernelWendlandC2_AVX256(dist, epsilon);
      /* accumulate into sum_w (contiguous) */
#if (PETSC_MEMALIGN == 32)
      sum_v = _mm256_load_pd(&sum_w[np]);
      sum_v = _mm256_add_pd(sum_v, rbf);
      _mm256_store_pd(&sum_w[np], sum_v);
#else
      sum_v = _mm256_loadu_pd(&sum_w[np]);
      sum_v = _mm256_add_pd(sum_v, rbf);
      _mm256_storeu_pd(&sum_w[np], sum_v);
#endif
      /* 
        store per-point weight for this partition (stride = npart).
        AVX2 has no scatter: write 4 scalars (fast enough compared to kernel) 
      */
      _mm256_store_pd(w_arr, rbf);
      /* weights layout: weights[np*npart + puidx] */
      for (i=0; i<4; i++) {
        PetscInt idx = np + i;
        weights[idx * npart + puidx] = (PetscReal)w_arr[i];
      }
    }
    /* Handle remaining points (not aligned to 4) */
    for (; np<npoints; np++) {
      PetscReal dist;
      dist = 0.0;
      for (d=0; d<dim; d++) { 
        dist += (pcoords[dim*np + d] - LA_coords[dim*puidx + d]) * (pcoords[dim*np + d] - LA_coords[dim*puidx + d]); 
      }
      dist = PetscSqrtReal(dist);
      /* Compute the weight */
      weights[np*npart + puidx] = KernelWendlandC2(dist,p->epsilon);
      sum_w[np] += weights[np*npart + puidx];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SwarmInterpSetWeights_AVX256_3D(
  PetscInt  npart,       // number of partitions
  Partition *partitions, // array of partitions
  PetscReal *LA_coords,  // partition centres coordinates
  PetscInt  npoints,     // number of interpolation points
  PetscReal *pcoords,    // interpolation points coordinates
  PetscReal *weights,    // output weights (size npoints*npart)
  PetscReal *sum_w       // output sum of weights (size npoints)
)
{
  PetscInt d,n,np,i,np_aligned;
  PetscInt dim = 3;
  PetscFunctionBeginUser;

  np_aligned = (npoints/4)*4;
  for (n=0; n<npart; n++) {
    Partition p = partitions[n]; // current partition
    PetscInt  puidx;
    __m256d   epsilon,centre_x,centre_y,centre_z;

    PetscCall(PartitionGetIndex(p,&puidx));
    epsilon   = _mm256_set1_pd(p->epsilon);
    centre_x  = _mm256_set1_pd(LA_coords[3*puidx    ]);
    centre_y  = _mm256_set1_pd(LA_coords[3*puidx + 1]);
    centre_z  = _mm256_set1_pd(LA_coords[3*puidx + 2]);
    
    for (np=0; np<np_aligned; np+=4) {
      double  w_arr[4] __attribute__((aligned(32)));
      __m256d x,y,z,dx,dy,dz,dist2,dist,rbf,sum_v;
      /* Store the coordinates */
      x = _mm256_set_pd(
        pcoords[ 3*(np+3) ], pcoords[ 3*(np+2) ],
        pcoords[ 3*(np+1) ], pcoords[ 3*np     ]
      );
      y = _mm256_set_pd(
        pcoords[ 3*(np+3) + 1 ], pcoords[ 3*(np+2) + 1 ],
        pcoords[ 3*(np+1) + 1 ], pcoords[ 3*np     + 1 ]
      );
      z = _mm256_set_pd(
        pcoords[ 3*(np+3) + 2 ], pcoords[ 3*(np+2) + 2 ],
        pcoords[ 3*(np+1) + 2 ], pcoords[ 3*np     + 2 ]
      );
      /* Compute the distance between the point and the partition centre */
      dx    = _mm256_sub_pd(x, centre_x); // dx = x - centre_x
      dy    = _mm256_sub_pd(y, centre_y); // dy = y - centre_y
      dz    = _mm256_sub_pd(z, centre_z); // dz = z - centre_z
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz))); // dx*dx + dy*dy + dz*dz
      dist  = _mm256_sqrt_pd(dist2);    // sqrt(dx*dx + dy*dy + dz*dz)
      rbf   = KernelWendlandC2_AVX256(dist, epsilon);
      /* accumulate into sum_w (contiguous) */
#if (PETSC_MEMALIGN == 32)
      sum_v = _mm256_load_pd(&sum_w[np]);
      sum_v = _mm256_add_pd(sum_v, rbf);
      _mm256_store_pd(&sum_w[np], sum_v);
#else
      sum_v = _mm256_loadu_pd(&sum_w[np]);
      sum_v = _mm256_add_pd(sum_v, rbf);
      _mm256_storeu_pd(&sum_w[np], sum_v);
#endif
      /* 
        store per-point weight for this partition (stride = npart).
        AVX2 has no scatter: write 4 scalars (fast enough compared to kernel) 
      */
      _mm256_store_pd(w_arr, rbf);
      /* weights layout: weights[np*npart + puidx] */
      for (i=0; i<4; i++) {
        PetscInt idx = np + i;
        weights[idx * npart + puidx] = (PetscReal)w_arr[i];
      }
    }
    /* Handle remaining points (not aligned to 4) */
    for (; np<npoints; np++) {
      PetscReal dist;
      dist = 0.0;
      for (d=0; d<dim; d++) { 
        dist += (pcoords[dim*np + d] - LA_coords[dim*puidx + d]) * (pcoords[dim*np + d] - LA_coords[dim*puidx + d]); 
      }
      dist = PetscSqrtReal(dist);
      /* Compute the weight */
      weights[np*npart + puidx] = KernelWendlandC2(dist,p->epsilon);
      sum_w[np] += weights[np*npart + puidx];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetWeights_AVX256(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DMSwarmCellDM    celldm;
  DM               das,cdas;
  Vec              coords;
  PetscReal        *pcoords,*weights,*LA_coords,*sum_w;
  PetscInt         np,n,i,dim,npart,npg,npoints,np_aligned,nn,puidx;
  const char       **coordFields;
  __m256d          zero_v;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));

  /* Get data structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Get the rank-local DMDA (overlapping between processes) */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  /* Get partitions centre coordinates (DMDA's nodes) */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  PetscCall(DMGetCoordinates(das,&coords));
  PetscCall(VecGetArray(coords,&LA_coords));
  /* Get the number of nodes in the local DMDA */
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Get the number of global partitions */
  PetscCall(PUGetNPartitionsGlobal(pu,&npg));
  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCheck(nn == npg, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of weigths must equal the number of partitions in the global mesh: %" PetscInt_FMT ", found: %" PetscInt_FMT "",npg,nn);
  PetscCall(PetscCalloc1(npoints,&sum_w));
  switch (dim) {
  case 2:
    PetscCall(SwarmInterpSetWeights_AVX256_2D(npart,partitions,LA_coords,npoints,pcoords,weights,sum_w));
    break;
  
  case 3:
    PetscCall(SwarmInterpSetWeights_AVX256_3D(npart,partitions,LA_coords,npoints,pcoords,weights,sum_w));
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX256 weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
  }
  /* Normalize weights using AVX256 vectorization over npoints */
  np_aligned = (npoints/4)*4;
  zero_v = _mm256_setzero_pd();
  for (n=0; n<npart; n++) {
    Partition p = partitions[n]; // current partition
    PetscCall(PartitionGetIndex(p,&puidx));
    for (np=0; np<np_aligned; np+=4) {
      double  norm_arr[4] __attribute__((aligned(32)));
      __m256d sum_v,w_v,mask,div_v,norm_v;

#if (PETSC_MEMALIGN == 32)
      sum_v = _mm256_load_pd(&sum_w[np]);
#else
      sum_v = _mm256_loadu_pd(&sum_w[np]);
#endif
      /* load 4 weights for this partition into vector */
      w_v = _mm256_set_pd(
        weights[(np + 3) * npart + puidx],
        weights[(np + 2) * npart + puidx],
        weights[(np + 1) * npart + puidx],
        weights[ np      * npart + puidx]
      );
      /* where sum_v > 0: w /= sum_v, else set 0 (avoid div by zero) */
      mask   = _mm256_cmp_pd(sum_v, zero_v, _CMP_GT_OQ);
      /* safe division: compute w_v / sum_v, but keep zero where mask false */
      div_v  = _mm256_div_pd(w_v, sum_v);
      norm_v = _mm256_blendv_pd(zero_v, div_v, mask);
      /* store normalized weights back (no scatter: extract and scalar-store) */
      _mm256_store_pd(norm_arr, norm_v);
      for (i=0; i<4; i++) {
        PetscInt idx = np + i;
        weights[idx * npart + puidx] = (PetscReal)norm_arr[i];
      }
    }
    /* Handle remaining points (not aligned to 4) */
    for (; np<npoints; np++) {
      if (sum_w[np] > 0.0) {
        weights[np*npart + puidx] /= sum_w[np];
      } else {
        weights[np*npart + puidx] = 0.0;
      }
    }
  }
  PetscCall(PetscFree(sum_w));
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coords,&LA_coords));

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode SwarmInterpSetWeights(LagRBFCtx ctx, DM sw)
{
  PetscFunctionBeginUser;

  /* For now we only support sequential */
#if defined(__AVX__)
  PetscCall(SwarmInterpSetWeights_AVX256(ctx,sw));
#else
  PetscCall(SwarmInterpSetWeights_Scalar(ctx,sw));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpAssignUniquePointIndex(DM sw, PetscInt start_idx, PetscInt end_idx)
{
  PetscInt    p,npoints,max_local,max,np_local,np_global;
  PetscInt    *pidx;
  PetscMPIInt rank;
  MPI_Comm    comm;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  PetscCall(PetscObjectGetComm((PetscObject)sw,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  //PetscCall(DMSwarmGetField(sw, "pidx", NULL, NULL, (void **)&pidx));
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pidx));
  /* find max pid presently in the system */
  max_local = 0;
  for (p=0; p<npoints; p++) {
    PetscInt mp_idx = pidx[p]; 
    if ( mp_idx > max_local ) { max_local = mp_idx; }
  }
  PetscCallMPI(MPI_Allreduce(&max_local,&max,1,MPI_LONG,MPI_MAX,comm));
  /*PetscPrintf(comm,"[[ %s ]] : max_pid = %" PetscInt_FMT "\n",PETSC_FUNCTION_NAME,max);*/
  max += 1;

  /* give particles a unique identifier */
  np_local = end_idx - start_idx;
  PetscCallMPI(MPI_Scan(&np_local,&np_global,1,MPI_LONG,MPI_SUM,comm));
  //printf("rank %d : np_local = %" PetscInt_FMT ", np_global = %" PetscInt_FMT "\n",(int)rank,np_local,np_global);
  for (p=start_idx; p<end_idx; p++) {
    pidx[p] = max + (np_global-np_local) + (long int)(p-start_idx);
    //printf("assigning %" PetscInt_FMT " -> pid = %" PetscInt_FMT " \n", p, pidx[p] );
  }
  //PetscCall(DMSwarmRestoreField(sw,"pidx",NULL,NULL,(void **)&pidx));
  PetscCall(DMSwarmRestoreField(sw,DMSwarmField_pid,NULL,NULL,(void **)&pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpSetUp(LagRBFCtx ctx, DM *sw)
{
  DM       swarm;
  PetscInt npoints;
  PetscFunctionBeginUser;

  PetscCall(SwarmInterpCreate(ctx,&swarm));
  if (ctx->restart) {
    //PetscCall(SwarmInterpLoadCoordinates(pu,swarm));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Restart not implemented yet for Swarm Interpolation");
  } else {
    PetscCall(SwarmInterpSetCoordinatesFromOptions(ctx->pu,swarm,INSERT_VALUES));
  }
  PetscCall(SwarmInterpSetWeights(ctx,swarm));
  PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
  PetscCall(SwarmInterpAssignUniquePointIndex(swarm,0,npoints));
  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SwarmInterpAddMarkersFromOptions(LagRBFCtx ctx, DM sw, InsertMode mode)
{
  PartitionOfUnity pu;
  PetscFunctionBeginUser;
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(SwarmInterpSetCoordinatesFromOptions(pu,sw,mode));
  PetscCall(SwarmInterpSetWeights(ctx,sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}