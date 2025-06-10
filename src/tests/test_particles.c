#include <petsc.h>
#include <petsc/private/dmdaimpl.h>

#define DEBUG
#define NSD 2

/*
You can use a k-d tree to store all points of A.

Iterate points b of B, and for each point - find the nearest point in A (let it be a) in the k-d tree. 
The point b should be included in the result if and only if the distance d(a,b) is smaller than r.

Complexity will be O(|B| * log(|A|) + |A|*log(|A|))
*/

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

static PetscErrorCode ViewElementToVertex(DM da)
{
  PetscInt iel,k,nel,nen;
  const PetscInt *elidx;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;

  PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank));

  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  for (iel=0; iel<nel; iel++) {
    PetscPrintf(PETSC_COMM_SELF,"Rank[%" PetscInt_FMT "]: Element %" PetscInt_FMT ":\n",rank,iel);
    for (k=0; k<nen; k++) {
      PetscPrintf(PETSC_COMM_SELF,"  Rank[%" PetscInt_FMT "]: Vertex [%" PetscInt_FMT "]: %" PetscInt_FMT "\n",rank,k,elidx[nen*iel + k]);
    }
  }


  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VertexToElementInverseMappingCreate(DM da, PetscInt **npv, PetscInt ***map)
{
  PetscInt iel,k,nel,nen,M,N,m,n,si,sj,nx,ny;
  PetscInt *nel_per_vertex;
  PetscInt **inverse_map;
  const PetscInt *elidx;
  PetscFunctionBeginUser;

  PetscCall(DMDAGetInfo(da,NULL,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));

  PetscCall(PetscCalloc1(nx*ny,&nel_per_vertex));
  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];
      nel_per_vertex[ nidx ]++; // count the number of elements that contain the vertex
    }
  }
  /* Allocate the inverse map */
  PetscCall(PetscMalloc1(nx*ny,&inverse_map));
  for (k=0; k<nx*ny; k++) {
    PetscCall(PetscMalloc1(nel_per_vertex[k],&inverse_map[k]));
  }
  /* Fill the inverse map */
  // reset the counter
  PetscMemzero(nel_per_vertex,nx*ny*sizeof(PetscInt));
  for (iel=0; iel<nel; iel++) {
    for (k=0; k<nen; k++) {
      PetscInt nidx = elidx[nen*iel + k];
      inverse_map[ nidx ][ nel_per_vertex[nidx] ] = iel; // store the element index
      nel_per_vertex[ nidx ]++; // increment the counter
    }
  }
  *npv = nel_per_vertex;
  *map = inverse_map;
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VertexToElementInverseMappingDestroy(DM da, PetscInt **npv, PetscInt ***map)
{
  PetscInt n,si,sj,nx,ny;
  PetscInt *nel_per_vertex = *npv;
  PetscInt **inverse_map   = *map;
  PetscFunctionBeginUser;
  
  if (!nel_per_vertex || !inverse_map) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  for (n=0; n<nx*ny; n++) {
    if (inverse_map[n]) {
      PetscCall(PetscFree(inverse_map[n]));
    }
  }
  PetscCall(PetscFree(inverse_map));
  PetscCall(PetscFree(nel_per_vertex));
  *npv = NULL;
  *map = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode PartitionGetPointsIdx(
  PetscInt *nel_per_vertex, // number of elements per vertex
  PetscInt **inverse_map, // inverse map from vertex to elements
  PetscInt puidx, // partition index
  DM swarm_interp,
  PetscInt *ninterp, // number of interpolation points in the partition
  PetscInt *interp_idx, // indices of the interpolation points in the partition
  DM swarm_data,
  PetscInt *ndata, // number of data points in the partition
  PetscInt *data_idx // indices of the data points in the partition
)
{
  PetscInt epv,iel;
  PetscFunctionBeginUser;

  // number of elements per vertex
  epv = nel_per_vertex[puidx];
  *ninterp = 0;
  *ndata   = 0;
  for (iel=0; iel<epv; iel++) {
    PetscInt data_ppc,interp_ppc,cell;
    PetscInt *data_pidlist,*interp_pidlist;
    
    cell = inverse_map[puidx][iel];
    /* Find data points */
    PetscCall(DMSwarmSortGetPointsPerCell(swarm_data,cell,&data_ppc,&data_pidlist));
    memcpy(&data_idx[*ndata],data_pidlist,data_ppc * sizeof(PetscInt));
    *ndata += data_ppc; // number of data points in the partition
    PetscCall(DMSwarmSortRestorePointsPerCell(swarm_data,cell,&data_ppc,&data_pidlist));
    /* Find interpolation points */
    PetscCall(DMSwarmSortGetPointsPerCell(swarm_interp,cell,&interp_ppc,&interp_pidlist));
    memcpy(&interp_idx[*ninterp],interp_pidlist,interp_ppc * sizeof(PetscInt));
    *ninterp += interp_ppc; // number of interpolation points in the partition
    PetscCall(DMSwarmSortRestorePointsPerCell(swarm_interp,cell,&interp_ppc,&interp_pidlist));
  }

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
#if 0
static PetscErrorCode ShowCellID(DM sw)
{
  DM             da;
  DMSwarmCellDM  celldm;
  MatStencil     node;
  PetscInt       npoints,nel,nen,iel,k;
  PetscInt       i,j,si,sj,nx,ny,vertex;
  const PetscInt *elidx;
  PetscMPIInt    rank,size;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMSwarmGetCellDMActive(sw,&celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm,&da));
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));


  PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank));
  PetscCall(MPI_Comm_size(PetscObjectComm((PetscObject)da),&size));
  
  PetscCall(DMDAGetElements(da,&nel,&nen,&elidx));
  /*
  for (iel=0; iel<nel; iel++) {
    PetscPrintf(PETSC_COMM_SELF,"rank[%" PetscInt_FMT "] Element %" PetscInt_FMT "\n",rank,iel);
    for (k=0; k<nen; k++) {
      PetscPrintf(PETSC_COMM_SELF,"    rank[%" PetscInt_FMT "] Vertex [%" PetscInt_FMT "]: %" PetscInt_FMT "\n",rank,k,elidx[nen*iel + k]);
    }
  }
  */
  PetscCall(DMDARestoreElements(da,&nel,&nen,&elidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
#if 0
static PetscErrorCode CreateKDTree_Swarm(DM sw, PetscKDTree *tree)
{
  DMSwarmCellDM celldm;   
  PetscReal     *coords;
  PetscInt      npoints, dim, Nfc;
  const char    **coordFields;
  PetscFunctionBeginUser;

  /* Get sizes */
  PetscCall(DMSwarmGetLocalSize(sw,&npoints));
  PetscCall(DMGetDimension(sw, &dim));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(PetscKDTreeCreate(npoints,dim,coords,PETSC_COPY_VALUES,10,tree));
  PetscCall(DMSwarmRestoreField(sw,coordFields[0],NULL,NULL,(void **)&coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
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
    //sum_w[nip] = 0.0; // shouldn't be needed because of Calloc1
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

static PetscErrorCode FindPointsInBall(
  PetscReal test_point[], 
  PetscReal points[],
  PetscInt  npoints, 
  PetscReal r2, 
  PetscInt  *nfound, 
  PetscInt  found_idx[]
)
{
  PetscInt np,nd,d;
  PetscFunctionBeginUser;
  np = 0;
  for (nd=0; nd<npoints; nd++) {
    PetscReal dist;

    dist = 0.0;
    for (d=0; d<NSD; d++) {
      dist += (points[NSD*nd + d] - test_point[d]) * (points[NSD*nd + d] - test_point[d]);
    }
    if (dist <= r2) {
#ifdef DEBUG
      PetscPrintf(PETSC_COMM_WORLD,"Found point %" PetscInt_FMT " (%g,%g)\n",nd,points[NSD*nd],points[NSD*nd + 1]);
#endif
      found_idx[np] = nd; // store the index of the found point
      np++;
    }
  }
  *nfound = np;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLocalCoefficients(
  MPI_Comm  comm, 
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
  PetscInt  i,j;
  PetscInt  *mat_idx;
  PetscReal *data_rbf,*rhs;
  
  PetscFunctionBeginUser;
  /* Create the matrix that will contain the RBF for the data points */
  // No idea if I should set the local size or the global size in parallel, anyway for now it is not parallel
  PetscCall(MatCreateDense(comm,PETSC_DECIDE,PETSC_DECIDE,npoints,npoints,NULL,&RBF_data));
  PetscCall(MatSetFromOptions(RBF_data));
  // Create vector for the right-hand side
  PetscCall(VecCreate(comm,&RHS));
  PetscCall(VecSetSizes(RHS,PETSC_DECIDE,npoints));
  PetscCall(VecSetFromOptions(RHS));
  PetscCall(VecZeroEntries(RHS));
  PetscCall(VecGetArray(RHS,&rhs));

  // Create the KSP solver
  PetscCall(KSPCreate(comm,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"rbf_"));
  PetscCall(KSPSetTolerances(ksp,1e-10,PETSC_CURRENT,PETSC_CURRENT,PETSC_CURRENT));
  PetscCall(KSPSetFromOptions(ksp));
  
  /* Set the matrix with the RBF values */
  PetscCall(PetscMalloc1(npoints*npoints,&data_rbf));
  PetscCall(PetscMalloc1(npoints,&mat_idx)); // I think we need a special case for parallelism here
  for (i=0; i<npoints; i++){
    mat_idx[i] = i;
    for (j=0; j<npoints; j++) {
      ContructRBFMatrix(i,j,npoints,epsilon,&data_coords[NSD*found_idx[i]],&data_coords[NSD*found_idx[j]],data_rbf);
      //PetscPrintf(PETSC_COMM_WORLD,"RBF[%d,%d] = %g\n",i,j,data_rbf[i*np + j]);
    }
    rhs[i] = fval[found_idx[i]]; // set the right-hand side with the values of the function at the data points
  }
  PetscCall(VecRestoreArray(RHS,&rhs));
  /* Set the matrix with the RBF values */
  PetscCall(MatSetValues(RBF_data,npoints,mat_idx,npoints,mat_idx,data_rbf,ADD_VALUES));
  PetscCall(MatAssemblyBegin(RBF_data,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RBF_data,MAT_FINAL_ASSEMBLY));

  PetscCall(KSPSetOperators(ksp,RBF_data,RBF_data));
  /* Solve the system */
  PetscCall(KSPSolve(ksp,RHS,X));

  /* Cleanup */
  PetscCall(MatDestroy(&RBF_data));
  PetscCall(VecDestroy(&RHS));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFree(data_rbf));
  PetscCall(PetscFree(mat_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RBFInterpolationLocal(
  MPI_Comm  comm,
  PetscInt  npart, // number of partitions
  PetscInt  puidx, // partition index
  PetscInt  ninterp, // number of interpolation points belonging to the partition
  PetscInt  interp_idx[], // indices of the interpolation points belonging to the partition
  PetscReal interp_coords[], // coordinates of the interpolation points
  PetscInt  ndata, // number of data points belonging to the partition
  PetscInt  data_idx[], // indices of the data points belonging to the partition
  PetscReal data_coords[], // coordinates of the data points
  PetscReal weights[], // weights for the interpolation points
  PetscReal epsilon,
  Vec       X, // This is the vector with the local coefficients
  Vec       F
)
{
  Mat       RBF_interp;
  Vec       Pj,W;
  PetscInt  i,j;
  PetscInt  *imat_rows,*imat_cols;
  PetscReal *LA_weights,*LA_Pj,*interp_rbf;
  PetscFunctionBeginUser;
  /* Matrix for the RBF(interp - data) */
  //PetscCall(MatCreate(comm,&RBF_interp));
  // Not sure if this should be a dense matrix or not, but if it sparse, I need to figure out where are the non-zero entries
  PetscCall(MatCreateDense(comm,PETSC_DECIDE,PETSC_DECIDE,ninterp,ndata,NULL,&RBF_interp)); // comm should = PETSC_COMM_SELF
  PetscCall(MatSetFromOptions(RBF_interp));
  /* Vector for the weights */
  PetscCall(VecCreate(comm,&W));
  PetscCall(VecSetSizes(W,PETSC_DECIDE,ninterp));
  PetscCall(VecSetFromOptions(W));
  PetscCall(VecZeroEntries(W)); // Not sure we need this
  PetscCall(VecGetArray(W,&LA_weights));

  PetscCall(PetscMalloc1(ninterp*ninterp,&interp_rbf));
  PetscCall(PetscMalloc1(ninterp,&imat_rows));
  PetscCall(PetscMalloc1(ndata,&imat_cols));
  for (i=0; i<ninterp; i++) {
    imat_rows[i] = i; // rows are the interpolation points
    for (j=0; j<ndata; j++) {
      imat_cols[j] = j; // cols are the data points
      ContructRBFMatrix(i,j,ndata,epsilon,&interp_coords[NSD*interp_idx[i]],&data_coords[NSD*data_idx[j]],interp_rbf);
    }
    LA_weights[i] = weights[interp_idx[i]*npart + puidx]; // set the weights for the interpolation points
  }
  PetscCall(VecRestoreArray(W,&LA_weights));
  /* Set the matrix with the RBF values */
  PetscCall(MatSetValues(RBF_interp,ninterp,imat_rows,ndata,imat_cols,interp_rbf,ADD_VALUES));
  PetscCall(MatAssemblyBegin(RBF_interp,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RBF_interp,MAT_FINAL_ASSEMBLY));

  VecCreate(comm,&Pj);
  VecSetSizes(Pj,PETSC_DECIDE,ninterp);
  VecSetFromOptions(Pj);
  /* Compute the local interpolation */
  PetscCall(MatMult(RBF_interp,X,Pj));
  /* Compute Pj * W and sum into the solution */
  PetscCall(VecPointwiseMult(Pj,Pj,W)); // weight the local interpolation
  PetscCall(VecGetArray(Pj,&LA_Pj)); // Do I need to convert Vec to array or is there a way to use Vec directly?
  PetscCall(VecSetValues(F,ninterp,interp_idx,LA_Pj,ADD_VALUES)); // sum the local interpolation into the global solution vector 
  PetscCall(VecAssemblyBegin(F));
  PetscCall(VecAssemblyEnd(F));
  PetscCall(VecRestoreArray(Pj,&LA_Pj));
  
  /* Cleanup */
  PetscCall(MatDestroy(&RBF_interp));
  PetscCall(VecDestroy(&W));
  PetscCall(VecDestroy(&Pj));
  PetscCall(PetscFree(interp_rbf));
  PetscCall(PetscFree(imat_rows));
  PetscCall(PetscFree(imat_cols));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RBFInterpolation(
  DM da, 
  DM swarm_data, 
  DM swarm_interp, 
  PetscReal radius,
  PetscReal epsilon
)
{
  DMSwarmCellDM dcelldm,icelldm;
  DM            cda;
  Vec           coord,F;
  PetscInt      i,j,si,sj,nx,ny,data_npoints,interp_npoints;
  PetscInt      *found_idx,*ifound_idx,*nel_per_vertex;
  PetscInt      **inverse_map;
  PetscReal     *LA_coords,*data_coords,*interp_coords;
  PetscReal     *franke,*weights;
  PetscReal     r2;
  const char    **dcoordFields,**icoordFields;
  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"[[ %s ]]\n",PETSC_FUNCTION_NAME);

  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  PetscCall(VertexToElementInverseMappingCreate(da,&nel_per_vertex,&inverse_map));
  /* Get partition of unity centres coords */
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&coord));
  PetscCall(VecGetArray(coord,&LA_coords));

  /* Data points */
  PetscCall(DMSwarmSortGetAccess(swarm_data));
  PetscCall(DMSwarmGetLocalSize(swarm_data,&data_npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_data, &dcelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(dcelldm, NULL, &dcoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_data, dcoordFields[0], NULL, NULL, (void **)&data_coords));
  PetscCall(DMSwarmGetField(swarm_data,"franke",NULL,NULL,(void **)&franke));

  /* Interpolation points */
  PetscCall(DMSwarmSortGetAccess(swarm_interp));
  PetscCall(DMSwarmGetLocalSize(swarm_interp,&interp_npoints));
  /* Setup for coords */
  PetscCall(DMSwarmGetCellDMActive(swarm_interp, &icelldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(icelldm, NULL, &icoordFields));
  /* Get fields */
  PetscCall(DMSwarmGetField(swarm_interp, icoordFields[0], NULL, NULL, (void **)&interp_coords));
  //PetscCall(DMSwarmGetField(swarm_interp,"franke",NULL,NULL,(void **)&interp_franke));
  PetscCall(DMSwarmGetField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  
  /* Create Vec to hold the interpolation */
  PetscCall(DMSwarmCreateGlobalVectorFromField(swarm_interp,"franke",&F));

  r2 = radius * radius; // squared radius for distance comparison

  PetscCall(PetscMalloc1(data_npoints,&found_idx)); // number of points => over allocated, need to find a better way to do this 
  PetscCall(PetscMalloc1(interp_npoints,&ifound_idx)); // number of points => over allocated, need to find a better way to do this

  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      Vec       X;
      PetscInt  np,npi,puidx;
      PetscInt  iel,epv; // elements per vertex
      PetscReal *pu_centre;
      
      // partition index
      puidx = i + j*nx;
      // partition centre
      pu_centre = &LA_coords[NSD*puidx];
      PetscCall(PartitionGetPointsIdx(nel_per_vertex,inverse_map,puidx,swarm_interp,&npi,ifound_idx,swarm_data,&np,found_idx));

      /* Search which data points are in the radius of the partition */
      // brute force radius search for now => see if using adjacent cells of the PU is ok or if kd-tree is needed
      //PetscCall(FindPointsInBall(pu_centre,data_coords,data_npoints,r2,&np,found_idx));
      
#ifdef DEBUG
      PetscPrintf(PETSC_COMM_WORLD,"Found %" PetscInt_FMT " points in Partition %" PetscInt_FMT "\n",np,puidx);
#endif
      if (np == 0) continue; // If no data points found, skip this partition

      // Create vector for the solution
      //PetscCall(VecCreate(PetscObjectComm((PetscObject)da),&X));
      PetscCall(VecCreate(PETSC_COMM_SELF,&X));
      PetscCall(VecSetSizes(X,PETSC_DECIDE,np));
      PetscCall(VecSetFromOptions(X));
      PetscCall(VecZeroEntries(X)); // Not sure we need this
      // Solve for local coefficients of the RBF interpolation
      PetscCall(SolveLocalCoefficients(PETSC_COMM_SELF,np,found_idx,data_coords,epsilon,franke,X));

      /* Search which interpolation points are in the radius of the partition */
      // brute force radius search for now
      //PetscCall(FindPointsInBall(pu_centre,interp_coords,interp_npoints,r2,&npi,ifound_idx));
#ifdef DEBUG
      PetscPrintf(PETSC_COMM_WORLD,"Found %" PetscInt_FMT " interpolation points in Partition %" PetscInt_FMT "\n",npi,puidx);
#endif
      PetscCall(RBFInterpolationLocal(PetscObjectComm((PetscObject)da),nx*ny,puidx,npi,ifound_idx,interp_coords,np,found_idx,data_coords,weights,epsilon,X,F));
      
      /* Cleanup */
      PetscCall(VecDestroy(&X));
    }
  }
  /* Restore and cleanup */
  PetscCall(DMSwarmSortRestoreAccess(swarm_data));
  PetscCall(DMSwarmRestoreField(swarm_data,dcoordFields[0],NULL,NULL,(void **)&data_coords));
  PetscCall(DMSwarmRestoreField(swarm_data,"franke",NULL,NULL,(void **)&franke));
  PetscCall(VecRestoreArray(coord,&LA_coords));
  PetscCall(PetscFree(found_idx));

  PetscCall(DMSwarmSortRestoreAccess(swarm_interp));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(swarm_interp,"franke",&F));
  PetscCall(DMSwarmRestoreField(swarm_interp,icoordFields[0],NULL,NULL,(void **)&interp_coords));
  //PetscCall(DMSwarmRestoreField(swarm_interp,"franke",NULL,NULL,(void **)&interp_franke));
  PetscCall(DMSwarmRestoreField(swarm_interp,"w_j",NULL,NULL,(void **)&weights));
  PetscCall(PetscFree(ifound_idx));

  PetscCall(VertexToElementInverseMappingDestroy(da,&nel_per_vertex,&inverse_map));

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

static PetscErrorCode TestMeshAttributes(void)
{
  DM          da;
  PetscInt    si,sj,nx,ny,m,n,M,N,dim,nr;
  const char  *prefix = "test_dmda";
  PetscMPIInt rank,size;
  MPI_Comm    comm;
  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;

  PetscCall(MPI_Comm_rank(comm,&rank));
  PetscCall(MPI_Comm_size(comm,&size));
  PetscPrintf(PETSC_COMM_WORLD,"Total cores: %" PetscInt_FMT "\n",size);

  /* Create the partition of unity mesh */
  PetscCall(DMDACreateMesh(comm, &da));
  PetscCall(DMDAGetInfo(da,&dim,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscPrintf(PETSC_COMM_WORLD,"Total number of partitions: %" PetscInt_FMT " x %" PetscInt_FMT " = %" PetscInt_FMT "\n",M,N,M*N);
  PetscPrintf(PETSC_COMM_WORLD,"Number of mpi processes per dimension: %" PetscInt_FMT " x %" PetscInt_FMT " = %" PetscInt_FMT "\n",m,n,m*n);
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  for (nr=0; nr<size; nr++) {
    if (rank == nr) {
      PetscPrintf(PETSC_COMM_SELF,"Rank: %" PetscInt_FMT ", Partitions: (%" PetscInt_FMT ",%" PetscInt_FMT ") x (%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,si,sj,si+nx,sj+ny);
    }
  }
  PetscCall(ViewElementToVertex(da));
  PetscCall(OutputDMDA(da, prefix));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestInterpolation(void)
{
  DM          da,swarm_data,swarm_interp;
  PetscReal   overlap,radius,epsilon;
  PetscInt    si,sj,nx,ny,m,n,M,N,dim,nr;
  const char  *prefix = "test_particles";
  PetscMPIInt rank,size;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_rank(comm,&rank));
  PetscCall(MPI_Comm_size(comm,&size));
  PetscPrintf(PETSC_COMM_WORLD,"Total cores: %" PetscInt_FMT "\n",size);

  /* Create the partition of unity mesh */
  PetscCall(DMDACreateMesh(comm, &da));
  PetscCall(DMDAGetInfo(da,&dim,&M,&N,NULL,&m,&n,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscPrintf(PETSC_COMM_WORLD,"Total number of partitions: %" PetscInt_FMT " x %" PetscInt_FMT " = %" PetscInt_FMT "\n",M,N,M*N);
  PetscPrintf(PETSC_COMM_WORLD,"Number of mpi processes per dimension: %" PetscInt_FMT " x %" PetscInt_FMT " = %" PetscInt_FMT "\n",m,n,m*n);
  PetscCall(DMDAGetCorners(da,&si,&sj,NULL,&nx,&ny,NULL));
  for (nr=0; nr<size; nr++) {
    if (rank == nr) {
      PetscPrintf(PETSC_COMM_SELF,"Rank: %" PetscInt_FMT ", Partitions: (%" PetscInt_FMT ",%" PetscInt_FMT ") x (%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,si,sj,si+nx,sj+ny);
    }
  }
  /* Create the swarm containing data */
  PetscCall(DMSwarmCreateDataPoints(da,&swarm_data));
  PetscCall(DMSwarmSetDataFunction(swarm_data));
  /* Create the swarm containing interpolation points */
  PetscCall(DMSwarmCreateInterpolationPoints(da,&swarm_interp));

  //PetscCall(ShowCellID(swarm_data));

  overlap = 1.5;
  radius  = overlap / PetscSqrtReal(M*N);
  epsilon = 1.0 / radius;
  PetscCall(PartitionOfUnity(da,swarm_data,swarm_interp,epsilon));
  PetscCall(RBFInterpolation(da,swarm_data,swarm_interp,radius,epsilon));
  /* Evaluate error */
  PetscCall(EvaluateError(swarm_interp));
  /* Output the mesh */
  PetscCall(OutputDMDA(da, prefix));
  /* output swarm data */
  {
    const char *fields_names[]= {"franke"};
    PetscCall(DMSwarmViewFieldsXDMF(swarm_data,"swarm_data.xmf",1,fields_names));
  }
  /* output interpolated data */
  {
    const char *fields_names[]= {"franke","w_j","error"};
    PetscCall(DMSwarmViewFieldsXDMF(swarm_interp,"swarm_interp.xmf",3,fields_names));
  }

  /* Clean up */
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&swarm_data));
  PetscCall(DMDestroy(&swarm_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscBool test = PETSC_FALSE;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test", &test, NULL));
  if (test) {
    PetscCall(TestMeshAttributes());
  } else {
    PetscCall(TestInterpolation());
  }
  
  PetscCall(PetscFinalize());
  
  return 0;
}