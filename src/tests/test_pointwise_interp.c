#include <petsc.h>
#include <petsc/private/dmdaimpl.h>

#define DEBUG
#define NSD 2

typedef struct {
  Vec       X; // Vector to hold the coefficients for the partition
  PetscInt  el_per_vertex; // Number of elements that contain the partition
  PetscInt  *map; // Inverse map from partition to elements
  PetscInt  ndata; // Number of data points in the partition
  PetscInt  *data_idx; // Indices of data points in the partition
  PetscReal epsilon; // Epsilon for the RBF kernel
} Partition;

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

static double WendlandC2(double r, double epsilon)
{
  /* Wendland C2 kernel */
  double u,value;
  
  u = 1.0 - epsilon * r;
  if (u < 0.0) return 0.0;
  value = pow(u,4) * (5.0 - 4.0 * u);
  return value;
}

static PetscErrorCode PartitionCreateVertexToElementInverseMapping(DM da, Partition **partition)
{
  PetscInt iel,k,nel,nen,M,N,m,n,si,sj,nx,ny;
  const PetscInt *elidx;
  PetscFunctionBeginUser;

  PetscCall(DMDAGetInfo(da,NULL,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));

  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];
      partition[nidx]->el_per_vertex++; // count the number of elements that contain the vertex
    }
  }
  
  for (k=0; k<nx*ny; k++) {
    /* Allocate the inverse map */
    PetscCall(PetscMalloc1(partition[k]->el_per_vertex,&(partition[k]->map)));
    /* Reset the counter */
    partition[k]->el_per_vertex = 0;
  }
  /* Fill the inverse map */
  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];

      partition[nidx]->map[ partition[nidx]->el_per_vertex ] = iel; // store the element index
      partition[nidx]->el_per_vertex++; // increment the counter
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionCreate(DM da, Partition ***partition)
{
  Partition **p;
  PetscInt si,sj,nx,ny;
  PetscFunctionBeginUser;

  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(PetscCalloc1(nx*ny,&p));
  for (int k=0; k<nx*ny; k++) {
    PetscCall(PetscNew(&p[k]));
    p[k]->el_per_vertex = 0; // initialize the number of elements per vertex
    p[k]->map           = NULL; // initialize the map to NULL
    p[k]->data_idx      = NULL; // initialize the data indices to NULL
    p[k]->epsilon       = 1.0; // default epsilon for the RBF kernel
    PetscCall(VecCreate(PETSC_COMM_SELF,&p[k]->X));
  }
  /* Get the inverse map connecting partition to cells */
  PetscCall(PartitionCreateVertexToElementInverseMapping(da,p));
  *partition = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionDestroy(DM da, Partition ***partition)
{
  PetscInt k,si,sj,nx,ny;
  Partition **p = *partition;
  PetscFunctionBeginUser;

  if (!p) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  for (k=0; k<nx*ny; k++) {
    if (p[k]) {
      if (p[k]->map) { PetscCall(PetscFree(p[k]->map)); }
      if (p[k]->data_idx) { PetscCall(PetscFree(p[k]->data_idx)); }
      if (p[k]->X) { PetscCall(VecDestroy(&(p[k]->X))); }
      PetscCall(PetscFree(p[k]));
    }
  }
  PetscCall(PetscFree(p));
  *partition = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ContructRBFMatrix(
  PetscInt row_idx,
  PetscInt col_idx,
  PetscInt ncols, 
  PetscReal epsilon,
  PetscReal point_A[],
  PetscReal point_B[],
  PetscReal rbf[]
)
{
  PetscInt  d;
  PetscReal dist;
  PetscFunctionBeginUser;
  dist = 0.0;
  for (d=0; d<NSD; d++) {
    dist += (point_A[d] - point_B[d]) * (point_A[d] - point_B[d]);
  }
  dist = PetscSqrtReal(dist);
  rbf[row_idx*ncols + col_idx] = WendlandC2(dist,epsilon);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OutputDMDA(DM da, const char prefix[])
{
  Vec         testvec;
  PetscViewer viewer;
  PetscBool   is_h5;
  char        fname[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
  is_h5 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-format_hdf5",&is_h5,NULL));

  if (is_h5) {
    PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s.h5", prefix);
    PetscCall(PetscViewerSetType(viewer,PETSCVIEWERHDF5));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_HDF5_PETSC));
  } else {
    PetscSNPrintf(fname, PETSC_MAX_PATH_LEN-1, "%s.vts", prefix);
    PetscCall(PetscViewerSetType(viewer,PETSCVIEWERVTK));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTS));
  }
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,fname));

  /* Create a vec just to print out the mesh */
  PetscCall(DMCreateGlobalVector(da,&testvec));
  PetscCall(VecSet(testvec,1.0));
  PetscCall(VecView(testvec,viewer));
  PetscCall(VecDestroy(&testvec));

  PetscCall(PetscViewerDestroy(&viewer));
  //PetscPrintf(PETSC_COMM_WORLD,"Wrote mesh to %s\n",fname);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDACreateMesh(MPI_Comm comm, DM *da)
{
  DM       dm;
  PetscInt m[] = {8, 8}, nm = 2;
  PetscReal O[] = {0.0, 0.0};
  PetscReal L[] = {1.0, 1.0};
  char      dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscBool found;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "pu_mesh");
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-m",m,&nm,&found));
  if (found) PetscCheck(nm == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-%sm must have 2 values, found: %" PetscInt_FMT "",dm_prefix,nm);
  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m[0],m[1],PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&dm));
  PetscObjectSetName((PetscObject)dm,dm_name);
  PetscCall(DMSetOptionsPrefix(dm,dm_prefix));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  /* Set coordinates */
  PetscCall(PetscOptionsGetRealArray(NULL,dm_prefix,"-origin",O,&nm,&found));
  if (found) PetscCheck(nm == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-dm_origin must have 2 values, found: %" PetscInt_FMT "", nm);
  PetscCall(PetscOptionsGetRealArray(NULL,dm_prefix,"-length",L,&nm,&found));
  if (found) PetscCheck(nm == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-dm_length must have 2 values, found: %" PetscInt_FMT "", nm);
  PetscCall(DMDASetUniformCoordinates(dm,O[0],L[0],O[1],L[1],0.0,0.0));

  *da = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmCreatePIC(DM da, DM *sw)
{
  DM       swarm;
  PetscInt dim;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMGetDimension(da, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)da), &swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscCall(DMSwarmSetCellDM(swarm, da));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmCreateDataPoints(DM da, DM *sw)
{
  DM        swarm;
  PetscInt  m[] = {16, 16}, ndim, d;
  PetscReal O[3],L[3];
  PetscBool found;
  char     dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmCreatePIC(da,&swarm));
  
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_data");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  /* create the field that will hold the data function values */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"franke", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  /* Set coords */
  DMGetDimension(swarm,&ndim);
  DMGetDimension(swarm,&d);
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-nppc",m,&d,&found));
  if (found) PetscCheck(d == ndim, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-%snppc must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "", dm_prefix, ndim, d);
  /* Create particles and set coordinates (all in one) */
  PetscCall(DMGetBoundingBox(da,O,L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmSetDataFunction(DM sw)
{
  DMSwarmCellDM celldm;
  PetscReal     *coords,*franke;
  PetscInt      Nfc,npoints,np;
  const char    **coordFields, *cellid;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  /* Get points coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,"franke",NULL,NULL,(void **)&franke));

  for (np=0; np<npoints; np++) {
    /* Get the coordinates of the particle */
    PetscReal x = coords[2*np];
    PetscReal y = coords[2*np+1];
    /* Compute the value of the function at that point */
    franke[np] = franke_function(x,y);
  }
  /* Set the field with the computed values */
  PetscCall(DMSwarmRestoreField(sw,"franke",NULL,NULL,(void **)&franke));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmCreateInterpolationPoints(DM da, DM *sw)
{
  DM        swarm;
  PetscInt  m[] = {16, 16}, ndim, d, si,sj,nx,ny;
  PetscReal O[3],L[3];
  PetscBool found;
  char     dm_name[PETSC_MAX_PATH_LEN],dm_prefix[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmCreatePIC(da,&swarm));
  /* Get number of partitions */
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  
  PetscSNPrintf(dm_name, PETSC_MAX_PATH_LEN-1, "swarm_interp");
  PetscObjectSetName((PetscObject)swarm,dm_name);
  PetscSNPrintf(dm_prefix, PETSC_MAX_PATH_LEN-1, "%s_",dm_name);
  PetscCall(DMSetOptionsPrefix(swarm,dm_prefix));
  PetscCall(DMSetFromOptions(swarm));
  /* create the field that will hold the data function values */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"franke", 1,     PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"w_j",    nx*ny, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"error",  1,     PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  /* Set coords */
  DMGetDimension(swarm,&ndim);
  DMGetDimension(swarm,&d);
  PetscCall(PetscOptionsGetIntArray(NULL,dm_prefix,"-nppc",m,&d,&found));
  if (found) PetscCheck(d == ndim, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "-%snppc must have %" PetscInt_FMT " values, found: %" PetscInt_FMT "", dm_prefix, ndim, d);
  /* Create particles and set coordinates (all in one) */
  PetscCall(DMGetBoundingBox(da,O,L));
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,O,L,m,INSERT_VALUES));

  *sw = swarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionOfUnity(DM da, DM swarm_data, DM swarm_interp, PetscReal epsilon)
{
  DMSwarmCellDM celldm;
  DM            cda;
  Vec           coord;
  PetscReal     *LA_coords,*icoords,*weights,*sum_w;
  PetscInt      i,j,si,sj,nx,ny,nip;
  PetscInt      np_interp,npu;
  const char    **icoordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  /* PU info */
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  
  /* Interpolation points info */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&np_interp)); // rank local points
  PetscCall(DMSwarmGetCellDMActive(swarm_interp, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &icoordFields));
  PetscCall(DMSwarmGetField(swarm_interp, icoordFields[0], NULL, NULL, (void **)&icoords));
  PetscCall(DMSwarmGetField(swarm_interp, "w_j", &npu, NULL, (void **)&weights));
  PetscCheck(npu == nx*ny, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of weigths must equal the number of partitions of unity, found: %" PetscInt_FMT "", npu);

  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords)); // I think this is rank local

  /* Sum for weights */
  PetscCall(PetscCalloc1(np_interp,&sum_w)); // local rank => Need to talk with Dave about parallelism
  /* Loop over points */
  for (nip=0; nip<np_interp; nip++) {
    /* Initialize sum of weights */
    /* Loop over partitions of unity */
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        PetscInt  d,puidx;
        PetscReal *pu_centre;
        double    dist=0.0;
        puidx = i + j*nx; // partition index, rank local (not the global index)
        /* Get the coordinates of the partition of unity centre */
        pu_centre = &LA_coords[2*puidx];
        for (d=0; d<2; d++) { dist += (icoords[2*nip+d] - pu_centre[d]) * (icoords[2*nip+d] - pu_centre[d]); }
        dist = PetscSqrtReal(dist);
        /* Compute the weight */
        weights[nip*nx*ny + puidx] = WendlandC2(dist,epsilon);
        sum_w[nip] += weights[nip*nx*ny + puidx];
      }
    }
  }
  // Here should perform communication to get the total sum of weights and send it to all processes 
  // but I am not sure if all processes have the same number of points so for now no parallelism
  /* Normalize weights */
  for (nip=0; nip<np_interp; nip++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        PetscInt puidx = i + j*nx;
        if (sum_w[nip] > 0.0) {
          /* Normalize the weights */
        weights[nip*nx*ny + puidx] /= sum_w[nip];
        } else {
          /* If the sum of weights is zero, set the weight to zero */
          weights[nip*nx*ny + puidx] = 0.0;
        }
      }
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swarm_interp,icoordFields[0],NULL,NULL,(void **)&icoords));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  /* Free memory */
  PetscCall(PetscFree(sum_w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblePartitionRBF(
  PetscInt npoints,
  PetscInt found_idx[],
  PetscReal data_coords[],
  PetscReal epsilon,
  Mat *RBF
)
{
  Mat       RBF_data;
  PetscReal *data_rbf;
  PetscInt  *mat_idx;
  PetscInt  i,j;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,npoints,npoints,NULL,&RBF_data));
  PetscCall(MatSetFromOptions(RBF_data));

  /* Set the matrix with the RBF values */
  PetscCall(PetscMalloc1(npoints*npoints,&data_rbf));
  PetscCall(PetscMalloc1(npoints,&mat_idx));
  for (i=0; i<npoints; i++){
    mat_idx[i] = i;
    for (j=0; j<npoints; j++) {
      ContructRBFMatrix(i,j,npoints,epsilon,&data_coords[NSD*found_idx[i]],&data_coords[NSD*found_idx[j]],data_rbf);
      //PetscPrintf(PETSC_COMM_WORLD,"RBF[%d,%d] = %g\n",i,j,data_rbf[i*np + j]);
    }
  }
  PetscCall(MatSetValues(RBF_data,npoints,mat_idx,npoints,mat_idx,data_rbf,ADD_VALUES));
  PetscCall(MatAssemblyBegin(RBF_data,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RBF_data,MAT_FINAL_ASSEMBLY));

  *RBF = RBF_data;
  /* Cleanup */
  PetscCall(PetscFree(data_rbf));
  PetscCall(PetscFree(mat_idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetRHS(
  PetscInt npoints,
  PetscInt found_idx[],
  PetscReal values[],
  Vec *B
)
{
  Vec       RHS;
  PetscInt  i;
  PetscReal *rhs;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(VecCreate(PETSC_COMM_SELF,&RHS));
  PetscCall(VecSetSizes(RHS,PETSC_DECIDE,npoints));
  PetscCall(VecSetFromOptions(RHS));
  PetscCall(VecGetArray(RHS,&rhs));
  /* set the right-hand side with the values of the function at the data points */
  for (i=0; i<npoints; i++){ rhs[i] = values[found_idx[i]]; }
  PetscCall(VecRestoreArray(RHS,&rhs));
  *B = RHS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PartitionGetPointsIdx(
  PetscInt nel_per_vertex, // number of elements per vertex
  PetscInt *inverse_map, // inverse map from vertex to elements
  DM swarm_data,
  PetscInt *ndata, // number of data points in the partition
  PetscInt *data_idx // indices of the data points in the partition
)
{
  PetscInt iel;
  PetscFunctionBeginUser;
  *ndata   = 0;
  for (iel=0; iel<nel_per_vertex; iel++) {
    PetscInt data_ppc,cell;
    PetscInt *data_pidlist;
    
    cell = inverse_map[iel];
    /* Find data points */
    PetscCall(DMSwarmSortGetPointsPerCell(swarm_data,cell,&data_ppc,&data_pidlist));
    memcpy(&data_idx[*ndata],data_pidlist,data_ppc * sizeof(PetscInt));
    *ndata += data_ppc; // number of data points in the partition
    PetscCall(DMSwarmSortRestorePointsPerCell(swarm_data,cell,&data_ppc,&data_pidlist));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLocalCoefficients(
  PetscInt  npoints, 
  PetscInt  found_idx[],
  PetscReal data_coords[],
  PetscReal epsilon, 
  PetscReal fval[],
  Vec X
)
{
  KSP       ksp;
  Mat       RBF_data;
  Vec       RHS;
  
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);
  // Create the KSP solver
  PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"rbf_"));
  PetscCall(KSPSetTolerances(ksp,1e-10,PETSC_CURRENT,PETSC_CURRENT,PETSC_CURRENT));
  PetscCall(KSPSetFromOptions(ksp));
  /* Create and assemble the RBF matrix */
  PetscCall(MatAssemblePartitionRBF(npoints,found_idx,data_coords,epsilon,&RBF_data));
  /* Create and fill the RHS vector */
  PetscCall(VecSetRHS(npoints,found_idx,fval,&RHS));
  /* Set operators for KSP */
  PetscCall(KSPSetOperators(ksp,RBF_data,RBF_data));
  /* Solve the system */
  PetscPrintf(PETSC_COMM_WORLD,"Solving local coefficients for %d points\n",npoints);
  PetscCall(KSPSolve(ksp,RHS,X));

  /* Cleanup */
  PetscCall(MatDestroy(&RBF_data));
  PetscCall(VecDestroy(&RHS));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveInterpolationCoefficients(
  Partition **partition,
  DM da,
  DM swarm_data
)
{
  DMSwarmCellDM celldm;
  DM            cda;
  Vec           coord;
  PetscInt      i,j,puidx,si,sj,nx,ny,npoints,nfound;
  PetscInt      *data_idx;
  PetscReal     *LA_coords,*data_coords,*franke;
  const char    **dcoordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  
  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords));

  /* Data points */
  PetscCall(DMSwarmSortGetAccess(swarm_data));
  PetscCall(DMSwarmGetLocalSize(swarm_data,&npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, NULL, &dcoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_data, dcoordFields[0], NULL, NULL, (void **)&data_coords));
  PetscCall(DMSwarmGetField(swarm_data,"franke",NULL,NULL,(void **)&franke));

  /* Allocate memory for the data indices */
  PetscCall(PetscMalloc1(npoints,&data_idx)); // we don't know how many points will be in each partition, so we allocate the maximum number of points

  /* Loop over partitions of unity */
  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      puidx = i + j*nx; // partition index, rank local (not the global index)

      /* Get the number of data points in the partition */
      PetscCall(PartitionGetPointsIdx(partition[puidx]->el_per_vertex,partition[puidx]->map,swarm_data,&nfound,data_idx));
      partition[puidx]->ndata = nfound; // store the number of data points in the partition
      PetscMalloc1(nfound,&(partition[puidx]->data_idx));
      /* Copy the data indices to the partition */
      memcpy(partition[puidx]->data_idx,data_idx,nfound * sizeof(PetscInt));
      if (nfound == 0) continue; // no data points in this partition, skip
      /* Set the size of the vector for the coefficients */
      PetscCall(VecSetSizes(partition[puidx]->X,PETSC_DECIDE,nfound));
      PetscCall(VecSetFromOptions(partition[puidx]->X));
      PetscCall(VecZeroEntries(partition[puidx]->X));
      PetscCall(SolveLocalCoefficients(nfound,data_idx,data_coords,partition[puidx]->epsilon,franke,partition[puidx]->X));
    }
  }
  /* Restore and cleanup */
  PetscCall(DMSwarmSortRestoreAccess(swarm_data));
  PetscCall(DMSwarmRestoreField(swarm_data,dcoordFields[0],NULL,NULL,(void **)&data_coords));
  PetscCall(DMSwarmRestoreField(swarm_data,"franke",NULL,NULL,(void **)&franke));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  PetscCall(PetscFree(data_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointwiseInterpolation(
  DM da, 
  DM swarm_data,
  DM swarm_interp,
  Partition **partition
)
{
  DMSwarmCellDM icelldm,dcelldm;
  DM            cda;
  Vec           coord;
  PetscInt      *interp_cellid;
  PetscInt      p,k,si,sj,nx,ny,nel,nen,interp_npoints;
  PetscReal     *interp_coords,*data_coords,*weights,*LA_coords,*interp_values;
  const PetscInt *elidx;
  const char    **icoordFields,**dcoordFields,*icellidFields;
  PetscFunctionBeginUser;

  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords));

  /* Interpolation points */
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&interp_npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_interp, &icelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(icelldm, NULL, &icoordFields));
  /* Setup for cell indices */
  PetscCall(DMSwarmCellDMGetCellID(icelldm,&icellidFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_interp, icoordFields[0], NULL, NULL, (void **)&interp_coords));
  PetscCall(DMSwarmGetField(swarm_interp, icellidFields, NULL, NULL, (void **)&interp_cellid));
  PetscCall(DMSwarmGetField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmGetField(swarm_interp,"franke",NULL,NULL,(void **)&interp_values));

  /* Data points */
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data, &dcelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(dcelldm, NULL, &dcoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_data, dcoordFields[0], NULL, NULL, (void **)&data_coords));

  for (p=0; p<interp_npoints; p++) {
    PetscInt cidx = interp_cellid[p]; // cell index for the particle p
    /* Initialize the interpolation value for the point p */
    interp_values[p] = 0.0;
    /* Iterate over the partitions to which that point is connected */
    for (k=0; k<nen; k++) {
      Vec       X;
      PetscInt  nd,puidx,ndata;
      PetscInt  *data_idx;
      PetscReal interpolant;
      PetscReal *coeffs;

      puidx    = elidx[nen*cidx + k];
      ndata    = partition[puidx]->ndata; // number of data points in the partition
      if (ndata == 0) continue; // no data points in this partition, skip
      data_idx = partition[puidx]->data_idx; // indices of the data points in the partition
      X        = partition[puidx]->X; // vector for the coefficients of the partition
      /* Get the coefficients of that partition */
      PetscCall(VecGetArray(X,&coeffs));
      interpolant = 0.0;
      /* Get the data points connected to that partition */
      for (nd=0; nd<ndata; nd++) {
        PetscInt  d;
        PetscReal dist,rbf;

        /* Compute the RBF value for the interpolation point */
        dist = 0.0;
        for (d=0; d<NSD; d++) {
          dist += (interp_coords[NSD*p + d] - data_coords[NSD*data_idx[nd] + d]) * (interp_coords[NSD*p + d] - data_coords[NSD*data_idx[nd] + d]);
        }
        dist = PetscSqrtReal(dist);
        rbf  = WendlandC2(dist,partition[puidx]->epsilon);
        rbf *= coeffs[nd]; // multiply the RBF value by the coefficient
        interpolant += rbf; // accumulate the value for the interpolation point
      }
      /* Normalize the interpolant with the weights */
      interp_values[p] += interpolant * weights[p*nx*ny + puidx];
      /* Restore the coefficients array */
      PetscCall(VecRestoreArray(X,&coeffs));
    }
  }
  PetscCall(DMSwarmRestoreField(swarm_interp,icoordFields[0],NULL,NULL,(void **)&interp_coords));
  PetscCall(DMSwarmRestoreField(swarm_interp,icellidFields,NULL,NULL,(void **)&interp_cellid));
  PetscCall(DMSwarmRestoreField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(DMSwarmRestoreField(swarm_interp,"franke",NULL,NULL,(void **)&interp_values));
  PetscCall(DMSwarmRestoreField(swarm_data,dcoordFields[0],NULL,NULL,(void **)&data_coords));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode EvaluateError(DM sw)
{
  DMSwarmCellDM celldm;
  PetscReal     sum,max_e;
  PetscReal     *coords,*franke,*error;
  PetscInt      Nfc,npoints,np;
  const char    **coordFields, *cellid;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  /* Get points coordinates */
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw,"franke",NULL,NULL,(void **)&franke));
  PetscCall(DMSwarmGetField(sw,"error",NULL,NULL,(void **)&error));

  sum = 0.0;
  max_e = PETSC_MIN_REAL;
  for (np=0; np<npoints; np++) {
    PetscReal diff;
    /* Get the coordinates of the particle */
    PetscReal x = coords[2*np];
    PetscReal y = coords[2*np+1];
    /* Compute the value of the function at that point */
    diff = PetscAbsReal(franke_function(x,y) - franke[np]);
    if (diff > max_e) max_e = diff;
    error[np] = diff / (PetscAbsReal(franke_function(x,y)) + PETSC_MACHINE_EPSILON);
    sum += diff*diff;
  }
  PetscPrintf(PETSC_COMM_WORLD,"L2 norm over the domain: %g\n",PetscSqrtReal(sum/npoints));
  PetscPrintf(PETSC_COMM_WORLD,"L_inf norm over the domain: %g\n",max_e);
  /* Set the field with the computed values */
  PetscCall(DMSwarmRestoreField(sw,"error",NULL,NULL,(void **)&error));
  PetscCall(DMSwarmRestoreField(sw,"franke",NULL,NULL,(void **)&franke));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
} 

static PetscErrorCode TestInterpolation()
{
  DM        da,swarm_data,swarm_interp;
  Partition **partition;
  PetscInt  k,si,sj,nx,ny,dim,M,N,m,n;
  PetscReal overlap,radius,epsilon;
  const char  *prefix = "test_ptwise";
  PetscFunctionBeginUser;

  PetscCall(DMDACreateMesh(PETSC_COMM_WORLD,&da));
  /* Create the swarm containing data */
  PetscCall(DMSwarmCreateDataPoints(da,&swarm_data));
  PetscCall(DMSwarmSetDataFunction(swarm_data));
  /* Create the swarm containing interpolation points */
  PetscCall(DMSwarmCreateInterpolationPoints(da,&swarm_interp));

  PetscCall(DMDAGetInfo(da,&dim,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(PartitionCreate(da,&partition));

  overlap = 1.5;
  radius  = overlap / PetscSqrtReal(M*N);
  epsilon = 1.0 / radius;
  for (k=0; k<nx*ny; k++) { partition[k]->epsilon = epsilon; } // set the epsilon for the RBF kernel
  /* Create the partition of unity (weights) */
  PetscCall(PartitionOfUnity(da,swarm_data,swarm_interp,epsilon));
  /* Solve for the partition-wise coefficients */
  PetscCall(SolveInterpolationCoefficients(partition,da,swarm_data));
  /* Perform interpolation */
  PetscCall(PointwiseInterpolation(da,swarm_data,swarm_interp,partition));
  /* Evaluate error */
  PetscCall(EvaluateError(swarm_interp));
  /* Output the mesh */
  PetscCall(OutputDMDA(da, prefix));
  /* output swarm data */
  {
    const char *fields_names[]= {"franke"};
    PetscCall(DMSwarmViewFieldsXDMF(swarm_data,"swarm_data_ptwise.xmf",1,fields_names));
  }
  /* output interpolated data */
  {
    const char *fields_names[]= {"franke","error"};
    PetscCall(DMSwarmViewFieldsXDMF(swarm_interp,"swarm_interp_ptwise.xmf",2,fields_names));
  }
  /* Clean-up */
  PetscCall(PartitionDestroy(da,&partition));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscCall(TestInterpolation());

  PetscCall(PetscFinalize());
  
  return 0;
}