#include <immintrin.h>
#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "advection/advection_ctx.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx512.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "cell_info.h"

PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX512;

static inline __m512d ComputeDistance_AVX512_2D(PetscReal *LA_coords, PetscInt *idx, __m512d *point)
{
  __m512d cx,cy,dx,dy,dist2,dist;
  /* Load 8 partition centers */
  cx = _mm512_set_pd(
    LA_coords[2*idx[7]    ],
    LA_coords[2*idx[6]    ],
    LA_coords[2*idx[5]    ],
    LA_coords[2*idx[4]    ],
    LA_coords[2*idx[3]    ],
    LA_coords[2*idx[2]    ],
    LA_coords[2*idx[1]    ],
    LA_coords[2*idx[0]    ]
  );
  cy = _mm512_set_pd(
    LA_coords[2*idx[7] + 1],
    LA_coords[2*idx[6] + 1],
    LA_coords[2*idx[5] + 1],
    LA_coords[2*idx[4] + 1],
    LA_coords[2*idx[3] + 1],
    LA_coords[2*idx[2] + 1],
    LA_coords[2*idx[1] + 1],
    LA_coords[2*idx[0] + 1]
  );
  
  /* Compute distance */
  dx    = _mm512_sub_pd(point[0], cx);
  dy    = _mm512_sub_pd(point[1], cy);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_mul_pd(dy, dy));
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

static inline __m512d ComputeDistance_AVX512_3D(PetscReal *LA_coords, PetscInt *idx, __m512d *point)
{
  __m512d cx,cy,cz,dx,dy,dz,dist2,dist;
  /* Load 8 partition centers */
  cx = _mm512_set_pd(
    LA_coords[3*idx[7]    ],
    LA_coords[3*idx[6]    ],
    LA_coords[3*idx[5]    ],
    LA_coords[3*idx[4]    ],
    LA_coords[3*idx[3]    ],
    LA_coords[3*idx[2]    ],
    LA_coords[3*idx[1]    ],
    LA_coords[3*idx[0]    ]
  );
  cy = _mm512_set_pd(
    LA_coords[3*idx[7] + 1],
    LA_coords[3*idx[6] + 1],
    LA_coords[3*idx[5] + 1],
    LA_coords[3*idx[4] + 1],
    LA_coords[3*idx[3] + 1],
    LA_coords[3*idx[2] + 1],
    LA_coords[3*idx[1] + 1],
    LA_coords[3*idx[0] + 1]
  );
  cz = _mm512_set_pd(
    LA_coords[3*idx[7] + 2],
    LA_coords[3*idx[6] + 2],
    LA_coords[3*idx[5] + 2],
    LA_coords[3*idx[4] + 2],
    LA_coords[3*idx[3] + 2],
    LA_coords[3*idx[2] + 2],
    LA_coords[3*idx[1] + 2],
    LA_coords[3*idx[0] + 2]
  );
  
  /* Compute distance */
  dx    = _mm512_sub_pd(point[0], cx);
  dy    = _mm512_sub_pd(point[1], cy);
  dz    = _mm512_sub_pd(point[2], cz);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode SwarmInterpSetWeights_AVX512(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DMSwarmCellDM    celldm;
  DM               das,cdas;
  Vec              coords;
  PetscReal        *pcoords,*weights,*LA_coords,*sum_w;
  PetscInt         d,k,n,np,dim,npart,npart_aligned,npoints,nn,nel,nen;
  PetscInt         *cellid,*gtol_elidx;
  const PetscInt   *elidx;
  const char       **coordFields,*cellFieldName;
  __m512d epsilon;
  __m512d (*compute_distance)(PetscReal *, PetscInt *, __m512d *);

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX512,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  /* Get data structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Get the rank-local DMDA (overlapping between processes) */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  /* Get partitions centre coordinates (DMDA's nodes) */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  PetscCall(DMGetCoordinates(das,&coords));
  PetscCall(VecGetArray(coords,&LA_coords));
  /* Get the number of nodes in the sequential DMDA */
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Get the parallel DMDA to sequential DMDA element map */
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Cell id */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellFieldName));
  PetscCall(DMSwarmGetField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCall(PetscCalloc1(npoints,&sum_w));
  
  switch (dim) {
  case 2:
    compute_distance = ComputeDistance_AVX512_2D;
    break;
  
  case 3:
    compute_distance = ComputeDistance_AVX512_3D;
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX512 weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
  }

  epsilon = _mm512_set1_pd(pu->epsilon_w);
  npart_aligned = (npart/8)*8; // number of partitions that can be processed with AVX512 (8 at a time)
  /* Process each point */
  for (np=0; np<npoints; np++) {
    PetscInt       eidx,eidx_local;
    const PetscInt *elem_nodes;
     __m512d       point[3];

    /* Cell index of the particle */
    eidx       = cellid[np];
    eidx_local = gtol_elidx[eidx];
    elem_nodes = &elidx[nen*eidx_local];
    
    for (d=0; d<dim; d++) { point[d] = _mm512_set1_pd(pcoords[dim*np + d]); }
    /* Loop over partitions 8 at a time using AVX512 */
    for (n=0; n<npart_aligned; n+=8) {
      PetscInt  puidx[8];
      double    w_arr[8] __attribute__((aligned(64)));
      __m512d   dist,rbf;

      /* Get partition indices */
      for (k=0; k<8; k++) { puidx[k] = partitions[n+k]->idx; }
      /* Compute distances */
      dist = compute_distance(LA_coords, puidx, point);
      /* Compute RBF weights */
      rbf = KernelWendlandC2_AVX512(dist, epsilon);
      /* Store weights */
      _mm512_store_pd(w_arr, rbf);
      
      /* Accumulate sum and store weights for nodes in this element */
      sum_w[np] += w_arr[0] + w_arr[1] + w_arr[2] + w_arr[3] + w_arr[4] + w_arr[5] + w_arr[6] + w_arr[7];
      
      /* Store weights - use if-else chain for faster search */
      for (k=0; k<nen; k++) {
        PetscInt nidx_e = elem_nodes[k];
        if      (nidx_e == puidx[0]) weights[np*nen + k] = (PetscReal)w_arr[0];
        else if (nidx_e == puidx[1]) weights[np*nen + k] = (PetscReal)w_arr[1];
        else if (nidx_e == puidx[2]) weights[np*nen + k] = (PetscReal)w_arr[2];
        else if (nidx_e == puidx[3]) weights[np*nen + k] = (PetscReal)w_arr[3];
        else if (nidx_e == puidx[4]) weights[np*nen + k] = (PetscReal)w_arr[4];
        else if (nidx_e == puidx[5]) weights[np*nen + k] = (PetscReal)w_arr[5];
        else if (nidx_e == puidx[6]) weights[np*nen + k] = (PetscReal)w_arr[6];
        else if (nidx_e == puidx[7]) weights[np*nen + k] = (PetscReal)w_arr[7];
      }
    }
    /* Handle remaining partitions (not aligned to 8) */
    for (; n<npart; n++) {
      PetscInt  puidx;
      PetscReal weight;
      PetscReal dist = 0.0;
      
      puidx = partitions[n]->idx;
      for (d=0; d<dim; d++) {
        dist += (pcoords[dim*np + d] - LA_coords[dim*puidx + d]) * (pcoords[dim*np + d] - LA_coords[dim*puidx + d]);
      }
      dist   = PetscSqrtReal(dist);
      weight = KernelWendlandC2(dist,pu->epsilon_w);
      sum_w[np] += weight;

      /* Store the weight only for the nodes connected to this element */
      for (k=0; k<nen; k++) {
        PetscInt nidx_e = elem_nodes[k];
        if (nidx_e == puidx) {
          weights[np*nen + k] = weight;
          break;
        }
      }
    }
  }

  /* Normalize weights */
  for (np=0; np<npoints; np++) {
    for (k=0; k<nen; k++) {
      if (sum_w[np] > 0.0) {
        weights[np*nen + k] /= sum_w[np];
      } else {
        weights[np*nen + k] = 0.0;
      }
    }
  }
  
  PetscCall(PetscFree(sum_w));
  PetscCall(DMSwarmRestoreField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coords,&LA_coords));

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX512,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Point-vectorized versions: compute distance from ONE partition to 8 points */
static inline __m512d ComputeDistance_AVX512_Points_2D(__m512d *cx, PetscReal *pcoords, PetscInt *pt_idx)
{
  __m512d px,py,dx,dy,dist2,dist;
  
  /* Load 8 x-coordinates from points */
  px = _mm512_set_pd(
    pcoords[2*pt_idx[7]    ],
    pcoords[2*pt_idx[6]    ],
    pcoords[2*pt_idx[5]    ],
    pcoords[2*pt_idx[4]    ],
    pcoords[2*pt_idx[3]    ],
    pcoords[2*pt_idx[2]    ],
    pcoords[2*pt_idx[1]    ],
    pcoords[2*pt_idx[0]    ]
  );
  
  /* Load 8 y-coordinates from points */
  py = _mm512_set_pd(
    pcoords[2*pt_idx[7] + 1],
    pcoords[2*pt_idx[6] + 1],
    pcoords[2*pt_idx[5] + 1],
    pcoords[2*pt_idx[4] + 1],
    pcoords[2*pt_idx[3] + 1],
    pcoords[2*pt_idx[2] + 1],
    pcoords[2*pt_idx[1] + 1],
    pcoords[2*pt_idx[0] + 1]
  );
  
  /* Compute distance */
  dx    = _mm512_sub_pd(px, cx[0]);
  dy    = _mm512_sub_pd(py, cx[1]);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_mul_pd(dy, dy));
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

static inline __m512d ComputeDistance_AVX512_Points_3D(__m512d *cx, PetscReal *pcoords, PetscInt *pt_idx)
{
  __m512d px,py,pz,dx,dy,dz,dist2,dist;
  
  /* Load 8 x-coordinates from points */
  px = _mm512_set_pd(
    pcoords[3*pt_idx[7]    ],
    pcoords[3*pt_idx[6]    ],
    pcoords[3*pt_idx[5]    ],
    pcoords[3*pt_idx[4]    ],
    pcoords[3*pt_idx[3]    ],
    pcoords[3*pt_idx[2]    ],
    pcoords[3*pt_idx[1]    ],
    pcoords[3*pt_idx[0]    ]
  );
  
  /* Load 8 y-coordinates from points */
  py = _mm512_set_pd(
    pcoords[3*pt_idx[7] + 1],
    pcoords[3*pt_idx[6] + 1],
    pcoords[3*pt_idx[5] + 1],
    pcoords[3*pt_idx[4] + 1],
    pcoords[3*pt_idx[3] + 1],
    pcoords[3*pt_idx[2] + 1],
    pcoords[3*pt_idx[1] + 1],
    pcoords[3*pt_idx[0] + 1]
  );
  
  /* Load 8 z-coordinates from points */
  pz = _mm512_set_pd(
    pcoords[3*pt_idx[7] + 2],
    pcoords[3*pt_idx[6] + 2],
    pcoords[3*pt_idx[5] + 2],
    pcoords[3*pt_idx[4] + 2],
    pcoords[3*pt_idx[3] + 2],
    pcoords[3*pt_idx[2] + 2],
    pcoords[3*pt_idx[1] + 2],
    pcoords[3*pt_idx[0] + 2]
  );
  
  /* Compute distance */
  dx    = _mm512_sub_pd(px, cx[0]);
  dy    = _mm512_sub_pd(py, cx[1]);
  dz    = _mm512_sub_pd(pz, cx[2]);
  dist2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
  dist  = _mm512_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode SwarmInterpSetWeights_AVX512_PointVectorized(LagRBFCtx ctx, DM sw)
{
  PartitionOfUnity pu;
  Partition        *partitions;
  DMSwarmCellDM    celldm;
  DM               das,cdas;
  Vec              coords;
  PetscReal        *pcoords,*weights,*LA_coords,*sum_w;
  PetscInt         d,k,m,n,np,dim,npart,npoints,npoints_aligned,nn,nel,nen;
  PetscInt         *cellid,*gtol_elidx;
  const PetscInt   *elidx;
  const char       **coordFields,*cellFieldName;
  __m512d epsilon;
  __m512d (*compute_distance)(__m512d *, PetscReal *, PetscInt *);

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX512,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  
  /* Get data structures */
  PetscCall(LagRBFGetPartitionOfUnity(ctx,&pu));
  PetscCall(LagRBFGetPartitions(ctx,&partitions));
  /* Get the rank-local DMDA (overlapping between processes) */
  PetscCall(PUGetDMDALocal(pu,&das));
  PetscCall(DMGetDimension(das,&dim));
  PetscCall(DMDAGetElements(das,&nel,&nen,&elidx));
  /* Get partitions centre coordinates (DMDA's nodes) */
  PetscCall(DMGetCoordinateDM(das,&cdas));
  PetscCall(DMGetCoordinates(das,&coords));
  PetscCall(VecGetArray(coords,&LA_coords));
  /* Get the number of nodes in the sequential DMDA */
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Get the parallel DMDA to sequential DMDA element map */
  PetscCall(PUGetGlobalToLocalElementMap(pu,&gtol_elidx));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  /* Setup for coords */
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&pcoords));
  /* Cell id */
  PetscCall(DMSwarmCellDMGetCellID(celldm,&cellFieldName));
  PetscCall(DMSwarmGetField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  /* Weights */
  PetscCall(DMSwarmGetField(sw, "weights", &nn, NULL, (void **)&weights));
  PetscCall(PetscCalloc1(npoints,&sum_w));
  
  switch (dim) {
  case 2:
    compute_distance = ComputeDistance_AVX512_Points_2D;
    break;
  
  case 3:
    compute_distance = ComputeDistance_AVX512_Points_3D;
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX512 point-vectorized weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
  }

  epsilon = _mm512_set1_pd(pu->epsilon_w);
  npoints_aligned = (npoints/8)*8; // number of points that can be processed with AVX512 (8 at a time)
  
  /* Process each partition */
  for (n=0; n<npart; n++) {
    PetscInt puidx = partitions[n]->idx;
    __m512d  centre[3];
    /* Load partition center into AVX512 register (reused for all 8 points) */
    for (d=0; d<dim; d++) {
      centre[d] = _mm512_set1_pd(LA_coords[dim*puidx + d]);
    }
    /* Process 8 points at a time using AVX512 */
    for (np=0; np<npoints_aligned; np+=8) {
      PetscInt  pt_idx[8];
      double    w_arr[8] __attribute__((aligned(64)));
      __m512d   dist,rbf;

      /* Set up point indices */
      for (k=0; k<8; k++) { pt_idx[k] = np + k; }
      
      /* Compute distances from partition to 8 points */
      dist = compute_distance(centre, pcoords, pt_idx);
      
      /* Compute RBF weights */
      rbf = KernelWendlandC2_AVX512(dist, epsilon);
      
      /* Store weights */
      _mm512_store_pd(w_arr, rbf);
      
      /* Vectorized sum_w update using AVX512 gather-add-scatter */
      {
        __m512i pt_idx_vec = _mm512_set_epi64(
          pt_idx[7], pt_idx[6], pt_idx[5], pt_idx[4],
          pt_idx[3], pt_idx[2], pt_idx[1], pt_idx[0]
        );
        __m512d sum_vals = _mm512_i64gather_pd(pt_idx_vec, sum_w, 8);
        sum_vals = _mm512_add_pd(sum_vals, rbf);
        _mm512_i64scatter_pd(sum_w, pt_idx_vec, sum_vals, 8);
      }
      
      /* Process all 8 points - element lookup cannot be easily vectorized due to data-dependent branching */
      for (k=0; k<8; k++) {
        PetscInt       pt = pt_idx[k];
        PetscInt       eidx,eidx_local;
        const PetscInt *elem_nodes;
        PetscReal      weight = (PetscReal)w_arr[k];
        
        /* Get element info for this point */
        eidx       = cellid[pt];
        eidx_local = gtol_elidx[eidx];
        elem_nodes = &elidx[nen*eidx_local];
        
        /* Store weight in element if partition is a node in this element */
        for (m=0; m<nen; m++) {
          if (elem_nodes[m] == puidx) {
            weights[pt*nen + m] = weight;
            break;
          }
        }
      }
    }
    
    /* Handle remaining points (not aligned to 8) */
    for (; np<npoints; np++) {
      PetscInt       eidx,eidx_local;
      const PetscInt *elem_nodes;
      PetscReal      dist = 0.0, weight;
      
      /* Compute distance */
      for (d=0; d<dim; d++) {
        dist += (pcoords[dim*np + d] - LA_coords[dim*puidx + d]) * (pcoords[dim*np + d] - LA_coords[dim*puidx + d]);
      }
      dist   = PetscSqrtReal(dist);
      weight = KernelWendlandC2(dist, pu->epsilon_w);
      sum_w[np] += weight;

      /* Get element info */
      eidx       = cellid[np];
      eidx_local = gtol_elidx[eidx];
      elem_nodes = &elidx[nen*eidx_local];
      
      /* Store the weight only for the nodes connected to this element */
      for (k=0; k<nen; k++) {
        if (elem_nodes[k] == puidx) {
          weights[np*nen + k] = weight;
          break;
        }
      }
    }
  }

  /* Normalize weights */
  for (np=0; np<npoints; np++) {
    for (k=0; k<nen; k++) {
      if (sum_w[np] > 0.0) {
        weights[np*nen + k] /= sum_w[np];
      } else {
        weights[np*nen + k] = 0.0;
      }
    }
  }
  
  PetscCall(PetscFree(sum_w));
  PetscCall(DMSwarmRestoreField(sw, cellFieldName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(sw,"weights",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&pcoords));
  PetscCall(VecRestoreArray(coords,&LA_coords));

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX512,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}