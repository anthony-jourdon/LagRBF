#ifndef softname_binary_loader_h
#define softname_binary_loader_h

typedef struct {
  int    n[3];
  int    nblocks;
  int    bs;
  double *data;
  float  *x,*y,*z;
  float  rcmb;
  bool   yy_reconstructed;
} StagFieldBin;

//PetscErrorCode StagBinSwarmSetFromFiles(DM swarm, const char **files_list, PetscInt nfiles, PetscBool scalars);
PetscErrorCode StagBinSwarmSetFromFiles(DM swarm, char **files_list, PetscInt nfiles, PetscBool *scalars);
#endif