#ifndef __softname_hdf5_read_fields_h
#define __softname_hdf5_read_fields_h

#include <hdf5.h>
#include <mpi.h>

struct _p_DatasetList {
  size_t count;
  size_t capacity;
  char   **names;
};
typedef struct _p_DatasetList  *DatasetList;

struct _p_StagFieldInfo {
  DatasetList datasets; // List of datasets for this field
  int         bs;       // blocksize of the data: scalars bs=1, vectors bs > 1
  int         npoints;  // total number of points
  double      *field;   // Data for the field
  void        *user;    // User defined context for flexibility
  void        (*destroy)(void*);
};
typedef struct _p_StagFieldInfo *StagFieldInfo;

void StagFieldInfoCreate(StagFieldInfo *field_data);
void StagFieldInfoDestroy(StagFieldInfo *field_data);
void HDF5_GetFieldShape(const char *filename, const char *dset_name, int *bs, int *npoints);
void HDF5_SetUpFieldData(MPI_Comm comm, const char *file_list[], int nfiles, StagFieldInfo *field_data);
void HDF5_GetFieldData(MPI_Comm comm, const char *file_list[], int nfiles, int step, StagFieldInfo field_data);
void HDF5_GetFieldFromFile(MPI_Comm comm, const char *filename, const char *dset_name, double *field);

#endif