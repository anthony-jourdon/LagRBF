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
  file:    weights_avx.c

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

#include <immintrin.h>
#include "lagrangian_rbf.h"
#include "partitions/partition.h"
#include "advection/advection_ctx.h"
#include "kernels/kernels_rbf.h"
#include "kernels/kernels_rbf_avx.h"
#include "swarm/swarm_utils.h"
#include "swarm/swarm_interp.h"
#include "cell_info.h"

PetscLogEvent LAGRBF_SwarmInterpSetWeights_AVX256;

static inline __m256d ComputeDistance_AVX256_2D(PetscReal *LA_coords, PetscInt *idx, __m256d *point)
{
  __m256d cx,cy,dx,dy,dist2,dist;
  /* Load 4 partition centers */
  cx = _mm256_set_pd(
    LA_coords[2*idx[3]    ],
    LA_coords[2*idx[2]    ],
    LA_coords[2*idx[1]    ],
    LA_coords[2*idx[0]    ]
  );
  cy = _mm256_set_pd(
    LA_coords[2*idx[3] + 1],
    LA_coords[2*idx[2] + 1],
    LA_coords[2*idx[1] + 1],
    LA_coords[2*idx[0] + 1]
  );
  
  /* Compute distance */
  dx    = _mm256_sub_pd(point[0], cx);
  dy    = _mm256_sub_pd(point[1], cy);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

static inline __m256d ComputeDistance_AVX256_3D(PetscReal *LA_coords, PetscInt *idx, __m256d *point)
{
  __m256d cx,cy,cz,dx,dy,dz,dist2,dist;
  /* Load 4 partition centers */
  cx = _mm256_set_pd(
    LA_coords[3*idx[3]    ],
    LA_coords[3*idx[2]    ],
    LA_coords[3*idx[1]    ],
    LA_coords[3*idx[0]    ]
  );
  cy = _mm256_set_pd(
    LA_coords[3*idx[3] + 1],
    LA_coords[3*idx[2] + 1],
    LA_coords[3*idx[1] + 1],
    LA_coords[3*idx[0] + 1]
  );
  cz = _mm256_set_pd(
    LA_coords[3*idx[3] + 2],
    LA_coords[3*idx[2] + 2],
    LA_coords[3*idx[1] + 2],
    LA_coords[3*idx[0] + 2]
  );
  
  /* Compute distance */
  dx    = _mm256_sub_pd(point[0], cx);
  dy    = _mm256_sub_pd(point[1], cy);
  dz    = _mm256_sub_pd(point[2], cz);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode SwarmInterpSetWeights_AVX256(LagRBFCtx ctx, DM sw)
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
  __m256d epsilon;
  __m256d (*compute_distance)(PetscReal *, PetscInt *, __m256d *);

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
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
    compute_distance = ComputeDistance_AVX256_2D;
    break;
  
  case 3:
    compute_distance = ComputeDistance_AVX256_3D;
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX256 weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
  }

  epsilon = _mm256_set1_pd(pu->epsilon_w);
  npart_aligned = (npart/4)*4; // number of partitions that can be processed with AVX (4 at a time)
  /* Process each point */
  for (np=0; np<npoints; np++) {
    PetscInt       eidx,eidx_local;
    const PetscInt *elem_nodes;
     __m256d       point[3];

    /* Cell index of the particle */
    eidx       = cellid[np];
    eidx_local = gtol_elidx[eidx];
    elem_nodes = &elidx[nen*eidx_local];
    
    for (d=0; d<dim; d++) { point[d] = _mm256_set1_pd(pcoords[dim*np + d]); }
    /* Loop over partitions 4 at a time using AVX */
    for (n=0; n<npart_aligned; n+=4) {
      PetscInt  puidx[4];
      double    w_arr[4] __attribute__((aligned(32)));
      __m256d   dist,rbf;

      /* Get partition indices */
      for (k=0; k<4; k++) { puidx[k] = partitions[n+k]->idx; }
      /* Compute distances */
      dist = compute_distance(LA_coords, puidx, point);
      /* Compute RBF weights */
      rbf = KernelWendlandC2_AVX256(dist, epsilon);
      /* Store weights */
      _mm256_store_pd(w_arr, rbf);
      
      /* Accumulate sum and store weights for nodes in this element */
      sum_w[np] += w_arr[0] + w_arr[1] + w_arr[2] + w_arr[3];
      
      /* Store weights - use if-else chain for faster search */
      for (k=0; k<nen; k++) {
        PetscInt nidx_e = elem_nodes[k];
        if      (nidx_e == puidx[0]) weights[np*nen + k] = (PetscReal)w_arr[0];
        else if (nidx_e == puidx[1]) weights[np*nen + k] = (PetscReal)w_arr[1];
        else if (nidx_e == puidx[2]) weights[np*nen + k] = (PetscReal)w_arr[2];
        else if (nidx_e == puidx[3]) weights[np*nen + k] = (PetscReal)w_arr[3];
      }
    }
    /* Handle remaining partitions (not aligned to 4) */
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

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline __m256d ComputeDistance_AVX256_Points_2D(__m256d *cx, PetscReal *pcoords, PetscInt *pt_idx)
{
  __m256d px,py,dx,dy,dist2,dist;
  
  /* Load 4 x-coordinates from points */
  px = _mm256_set_pd(
    pcoords[2*pt_idx[3]    ],
    pcoords[2*pt_idx[2]    ],
    pcoords[2*pt_idx[1]    ],
    pcoords[2*pt_idx[0]    ]
  );
  py = _mm256_set_pd(
    pcoords[2*pt_idx[3] + 1],
    pcoords[2*pt_idx[2] + 1],
    pcoords[2*pt_idx[1] + 1],
    pcoords[2*pt_idx[0] + 1]
  );
  
  /* Compute distances */
  dx    = _mm256_sub_pd(px, cx[0]);
  dy    = _mm256_sub_pd(py, cx[1]);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

static inline __m256d ComputeDistance_AVX256_Points_3D(__m256d *cx, PetscReal *pcoords, PetscInt *pt_idx)
{
  __m256d px,py,pz,dx,dy,dz,dist2,dist;
  
  /* Load 4 x-coordinates from points */
  px = _mm256_set_pd(
    pcoords[3*pt_idx[3]    ],
    pcoords[3*pt_idx[2]    ],
    pcoords[3*pt_idx[1]    ],
    pcoords[3*pt_idx[0]    ]
  );
  py = _mm256_set_pd(
    pcoords[3*pt_idx[3] + 1],
    pcoords[3*pt_idx[2] + 1],
    pcoords[3*pt_idx[1] + 1],
    pcoords[3*pt_idx[0] + 1]
  );
  pz = _mm256_set_pd(
    pcoords[3*pt_idx[3] + 2],
    pcoords[3*pt_idx[2] + 2],
    pcoords[3*pt_idx[1] + 2],
    pcoords[3*pt_idx[0] + 2]
  );
  
  /* Compute distances */
  dx    = _mm256_sub_pd(px, cx[0]);
  dy    = _mm256_sub_pd(py, cx[1]);
  dz    = _mm256_sub_pd(pz, cx[2]);
  dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
  dist  = _mm256_sqrt_pd(dist2);
  return dist;
}

PetscErrorCode SwarmInterpSetWeights_AVX256_PointVectorized(LagRBFCtx ctx, DM sw)
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
  __m256d epsilon;
  __m256d (*compute_distance)(__m256d *, PetscReal *, PetscInt *);

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
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
    compute_distance = ComputeDistance_AVX256_Points_2D;
    break;
  
  case 3:
    compute_distance = ComputeDistance_AVX256_Points_3D;
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX256 point-vectorized weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
  }

  epsilon = _mm256_set1_pd(pu->epsilon_w);
  npoints_aligned = (npoints/4)*4; // number of points that can be processed with AVX256 (4 at a time)
  /* Process each partition */
  for (n=0; n<npart; n++) {
    PetscInt puidx = partitions[n]->idx;
    __m256d  centre[3];
    /* Load partition center into AVX256 register (reused for all 4 points) */
    for (d=0; d<dim; d++) {
      centre[d] = _mm256_set1_pd(LA_coords[dim*puidx + d]);
    }
    /* Process 4 points at a time using AVX256 */
    for (np=0; np<npoints_aligned; np+=4) {
      PetscInt  pt_idx[4];
      double    w_arr[4] __attribute__((aligned(32)));
      __m256d   dist,rbf;

      /* Set up point indices */
      for (k=0; k<4; k++) { pt_idx[k] = np + k; }
      
      /* Compute distances from partition to 4 points */
      dist = compute_distance(centre, pcoords, pt_idx);
      
      /* Compute RBF weights */
      rbf = KernelWendlandC2_AVX256(dist, epsilon);
      
      /* Store weights */
      _mm256_store_pd(w_arr, rbf);
      
      /* Process all 4 points */
      for (k=0; k<4; k++) {
        PetscInt       pt = pt_idx[k];
        PetscInt       eidx,eidx_local;
        const PetscInt *elem_nodes;
        PetscReal      weight = (PetscReal)w_arr[k];
        
        sum_w[pt] += weight;
        
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
    
    /* Handle remaining points (not aligned to 4) */
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

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


#if 0
static PetscErrorCode SwarmInterpSetWeights_AVX256_2D_Opt(
  PartitionOfUnity pu,
  Partition        *partitions,
  PetscReal        *LA_coords,
  PetscInt         npoints,
  PetscReal        *pcoords,
  PetscInt         *cellid,
  PetscInt         *gtol_elidx,
  const PetscInt   *elidx,
  PetscInt         nen,
  PetscReal        *weights,
  PetscReal        *sum_w
)
{
  PetscInt d,n,np,k,npart;
  PetscInt dim = 2;
  PetscFunctionBeginUser;
  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  
  /* Process each point */
  for (np=0; np<npoints; np++) {
    PetscInt  eidx,eidx_local;
    __m256d   epsilon,pt_x,pt_y;
    const PetscInt *elem_nodes;

    /* Cell index of the particle */
    eidx       = cellid[np];
    eidx_local = gtol_elidx[eidx];
    elem_nodes = &elidx[nen*eidx_local];
    
    epsilon = _mm256_set1_pd(pu->epsilon);
    pt_x    = _mm256_set1_pd(pcoords[2*np    ]);
    pt_y    = _mm256_set1_pd(pcoords[2*np + 1]);
    
    /* Loop over partitions 4 at a time using AVX */
    for (n=0; n<(npart/4)*4; n+=4) {
      double    w_arr[4] __attribute__((aligned(32)));
      __m256d   cx,cy,dx,dy,dist2,dist,rbf;
      PetscInt  puidx0,puidx1,puidx2,puidx3;
      
      /* Get partition indices */
      puidx0 = partitions[n]->idx;
      puidx1 = partitions[n+1]->idx;
      puidx2 = partitions[n+2]->idx;
      puidx3 = partitions[n+3]->idx;
      
      /* Load 4 partition centers */
      cx = _mm256_set_pd(
        LA_coords[2*puidx3    ],
        LA_coords[2*puidx2    ],
        LA_coords[2*puidx1    ],
        LA_coords[2*puidx0    ]
      );
      cy = _mm256_set_pd(
        LA_coords[2*puidx3 + 1],
        LA_coords[2*puidx2 + 1],
        LA_coords[2*puidx1 + 1],
        LA_coords[2*puidx0 + 1]
      );
      
      /* Compute distances and weights */
      dx    = _mm256_sub_pd(pt_x, cx);
      dy    = _mm256_sub_pd(pt_y, cy);
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
      dist  = _mm256_sqrt_pd(dist2);
      rbf   = KernelWendlandC2_AVX256(dist, epsilon);
      
      /* Store weights */
      _mm256_store_pd(w_arr, rbf);
      
      /* Accumulate sum and store weights for nodes in this element */
      sum_w[np] += w_arr[0] + w_arr[1] + w_arr[2] + w_arr[3];
      
      /* Store weights - use if-else chain for faster search */
      for (k=0; k<nen; k++) {
        PetscInt nidx_e = elem_nodes[k];
        if (nidx_e == puidx0) weights[np*nen + k] = (PetscReal)w_arr[0];
        else if (nidx_e == puidx1) weights[np*nen + k] = (PetscReal)w_arr[1];
        else if (nidx_e == puidx2) weights[np*nen + k] = (PetscReal)w_arr[2];
        else if (nidx_e == puidx3) weights[np*nen + k] = (PetscReal)w_arr[3];
      }
    }
    
    /* Handle remaining partitions (not aligned to 4) */
    for (; n<npart; n++) {
      PetscInt  puidx;
      PetscReal dist,weight,dx,dy;
      
      puidx = partitions[n]->idx;
      dx = pcoords[2*np    ] - LA_coords[2*puidx    ];
      dy = pcoords[2*np + 1] - LA_coords[2*puidx + 1];
      dist = PetscSqrtReal(dx*dx + dy*dy);
      weight = KernelWendlandC2(dist,pu->epsilon);
      sum_w[np] += weight;

      /* Store the weight only for the nodes connected to this element */
      for (k=0; k<nen; k++) {
        if (elem_nodes[k] == puidx) {
          weights[np*nen + k] = weight;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Optimized version with pre-extracted indices and improved memory access */
static PetscErrorCode SwarmInterpSetWeights_AVX256_3D_Opt(
  PartitionOfUnity pu,
  Partition        *partitions,
  PetscReal        *LA_coords,
  PetscInt         npoints,
  PetscReal        *pcoords,
  PetscInt         *cellid,
  PetscInt         *gtol_elidx,
  const PetscInt   *elidx,
  PetscInt         nen,
  PetscReal        *weights,
  PetscReal        *sum_w
)
{
  PetscInt d,n,np,k,npart;
  PetscInt dim = 3;
  PetscFunctionBeginUser;

  PetscCall(PUGetNPartitionsLocal(pu,&npart));
  /* Process each point */
  for (np=0; np<npoints; np++) {
    PetscInt  eidx,eidx_local;
    __m256d   epsilon,pt_x,pt_y,pt_z;
    const PetscInt *elem_nodes;

    /* Cell index of the particle */
    eidx       = cellid[np];
    eidx_local = gtol_elidx[eidx];
    elem_nodes = &elidx[nen*eidx_local];
    
    epsilon = _mm256_set1_pd(pu->epsilon);
    pt_x    = _mm256_set1_pd(pcoords[3*np    ]);
    pt_y    = _mm256_set1_pd(pcoords[3*np + 1]);
    pt_z    = _mm256_set1_pd(pcoords[3*np + 2]);
    
    /* Loop over partitions 4 at a time using AVX */
    for (n=0; n<(npart/4)*4; n+=4) {
      double    w_arr[4] __attribute__((aligned(32)));
      __m256d   cx,cy,cz,dx,dy,dz,dist2,dist,rbf;
      PetscInt  puidx0,puidx1,puidx2,puidx3;
      
      /* Get partition indices */
      puidx0 = partitions[n]->idx;
      puidx1 = partitions[n+1]->idx;
      puidx2 = partitions[n+2]->idx;
      puidx3 = partitions[n+3]->idx;
      
      /* Load 4 partition centers */
      cx = _mm256_set_pd(
        LA_coords[3*puidx3    ],
        LA_coords[3*puidx2    ],
        LA_coords[3*puidx1    ],
        LA_coords[3*puidx0    ]
      );
      cy = _mm256_set_pd(
        LA_coords[3*puidx3 + 1],
        LA_coords[3*puidx2 + 1],
        LA_coords[3*puidx1 + 1],
        LA_coords[3*puidx0 + 1]
      );
      cz = _mm256_set_pd(
        LA_coords[3*puidx3 + 2],
        LA_coords[3*puidx2 + 2],
        LA_coords[3*puidx1 + 2],
        LA_coords[3*puidx0 + 2]
      );
      
      /* Compute distances and weights */
      dx    = _mm256_sub_pd(pt_x, cx);
      dy    = _mm256_sub_pd(pt_y, cy);
      dz    = _mm256_sub_pd(pt_z, cz);
      dist2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
      dist  = _mm256_sqrt_pd(dist2);
      rbf   = KernelWendlandC2_AVX256(dist, epsilon);
      
      /* Store weights */
      _mm256_store_pd(w_arr, rbf);
      
      /* Accumulate sum and store weights for nodes in this element */
      sum_w[np] += w_arr[0] + w_arr[1] + w_arr[2] + w_arr[3];
      
      /* Store weights - use if-else chain for faster search */
      for (k=0; k<nen; k++) {
        PetscInt nidx_e = elem_nodes[k];
        if      (nidx_e == puidx0) weights[np*nen + k] = (PetscReal)w_arr[0];
        else if (nidx_e == puidx1) weights[np*nen + k] = (PetscReal)w_arr[1];
        else if (nidx_e == puidx2) weights[np*nen + k] = (PetscReal)w_arr[2];
        else if (nidx_e == puidx3) weights[np*nen + k] = (PetscReal)w_arr[3];
      }
    }
    
    /* Handle remaining partitions (not aligned to 4) */
    for (; n<npart; n++) {
      PetscInt  puidx;
      PetscReal dist,weight,dx,dy,dz;
      
      puidx = partitions[n]->idx;
      dx = pcoords[3*np    ] - LA_coords[3*puidx    ];
      dy = pcoords[3*np + 1] - LA_coords[3*puidx + 1];
      dz = pcoords[3*np + 2] - LA_coords[3*puidx + 2];
      dist = PetscSqrtReal(dx*dx + dy*dy + dz*dz);
      weight = KernelWendlandC2(dist,pu->epsilon);
      sum_w[np] += weight;

      /* Store the weight only for the nodes connected to this element */
      for (k=0; k<nen; k++) {
        if (elem_nodes[k] == puidx) {
          weights[np*nen + k] = weight;
          break;
        }
      }
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
  PetscInt         k,np,dim,npart,npoints,nn,nel,nen;
  PetscInt         *cellid,*gtol_elidx;
  const PetscInt   *elidx;
  const char       **coordFields,*cellFieldName;
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s Optimized v1 ]]\n",PETSC_FUNCTION_NAME);
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
    PetscCall(SwarmInterpSetWeights_AVX256_2D_Opt(pu,partitions,LA_coords,npoints,pcoords,cellid,gtol_elidx,elidx,nen,weights,sum_w));
    break;
  
  case 3:
    PetscCall(SwarmInterpSetWeights_AVX256_3D_Opt(pu,partitions,LA_coords,npoints,pcoords,cellid,gtol_elidx,elidx,nen,weights,sum_w));
    break;
  
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"AVX256 weights only implemented for 2D and 3D, found dim = %" PetscInt_FMT "",dim);
    break;
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

  PetscCall(PetscLogEventEnd(LAGRBF_SwarmInterpSetWeights_AVX256,0,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif