import os
import numpy as np
import pyvista as pvs
import h5py
from .timeseries_filters import XDMFDataFilter
from ..petscvec_writer import PetscVecWrite

class StagYY:
  def __init__(self, root:str='.'):
    self.root = root
    return
  
class StagYYh5(StagYY):
  def __init__(self,xdmfile:str, root:str='.', output_dir:str=None):
    super().__init__(root)
    try:
      self.xdmfdata = XDMFDataFilter(xdmfile)
    except Exception as e:
      raise ValueError(f"Failed to process XDMF file {xdmfile}: {e}")
    self.mesh = None
    self.dim  = None
    if output_dir is None:
      self.output_dir = root

  def mesh_from_file(self, fname:str) -> pvs.StructuredGrid:
    if self.mesh is not None:
      return self.mesh
    with h5py.File(fname, 'r') as fp: # read hdf5 file
      entries = list(fp.keys())  # get all entries in the file
      shape = fp[entries[0]].shape
      coor  = np.zeros((shape[0]*shape[1]*shape[2], 3), dtype=np.float64)
      if shape[0] == 1: # 2D mesh
        permutation = {0:1, 1:2, 2:0} # Y-Z plane, X is 0
        self.dim = 2
      else: # 3D mesh
        permutation = {0:0, 1:1, 2:2} # X-Y-Z order
        self.dim = 3
      for i in range(3):
        coor[:,i] = fp[ entries[ permutation[i] ] ][:,:,:].reshape(shape[0]*shape[1]*shape[2])
      mesh = pvs.StructuredGrid(coor)
      mesh.dimensions = (shape[ permutation[0] ], shape[ permutation[1] ], shape[ permutation[2] ])
    return mesh
  
  def create_mesh(self) -> None:
    coor_fname = None
    filenames  = self.xdmfdata.get_filenames()
    coor_name  = 'NodeCoordinates'
    try:
      for fields in filenames:
        if coor_name in fields:
          coor_fname = fields
          break
      self.mesh = self.mesh_from_file(os.path.join(self.root, coor_fname))
    except Exception as e:
      raise ValueError(f"Failed to create mesh from file: {e}")
    return

  def datafield_from_file(self, fname:str, step:int=0):
    with h5py.File(fname, 'r') as fp:
      entries = list(fp.keys())
      dataset = fp[entries[step]]
      npoints = dataset.shape[0] * dataset.shape[1] * dataset.shape[2]
      # get block size
      if dataset.ndim > 3:
        bs = dataset.shape[3]
        data = np.zeros((npoints, bs), dtype=dataset.dtype)
        for b in range(bs):
          data[:,b] = dataset[:,:,:,b].reshape((npoints))
      else:
        data = dataset[:,:,:].reshape((npoints))
    return data
  
  def write_coords_petsc_vec(self, fname:str) -> None:
    """Write coordinates to a binary file using PETSc Vec data type."""
    if self.mesh is None:
      raise ValueError("Mesh has not been created. Call create_mesh() first.")
    if self.dim == 2:
      coor = self.mesh.points[:,[0,1]]
    elif self.dim == 3:
      coor = self.mesh.points
    else:
      raise ValueError(f"Unsupported mesh dimension. Only 2D and 3D meshes are supported. Found: {self.dim}")
    PetscVecWrite(fname, coor)
    return
  
  def write_field_petsc_vec(self, fname:str, field_name:str) -> None:
    """Write a field to a binary file using PETSc Vec data type."""
    if field_name not in self.mesh.point_data:
      raise ValueError(f"Field '{field_name}' not found in mesh point data. Check that the field exists or that it is a point data field.")
    PetscVecWrite(fname, self.mesh.point_data[field_name])
    return
  
  def add_fields_to_mesh(self, fields:list|None=None, step:int=0) -> None:
    if fields is None:
      fields = self.xdmfdata.fieldnames
    for field_name in fields:
      for filename in self.xdmfdata.filenames:
        if field_name in filename:
          try:
            self.mesh[field_name] = self.datafield_from_file(os.path.join(self.root, filename), step=step)
          except Exception as e:
            raise ValueError(f"Failed to add field '{field_name}' from file '{filename}': {e}")
    return

  def write_step_petsc_vec(self, fields:list|None=None, step:int=0):
    output_dir = os.path.join(self.output_dir, f"step{step}")
    os.makedirs(output_dir, exist_ok=True)
    if fields is None:
      fields = self.xdmfdata.fieldnames
    for field_name in fields:
      self.write_field_petsc_vec(os.path.join(output_dir, f"step{str(step).zfill(6)}.{field_name}.pbvec"), field_name)
    return

  def write_timeseries_petsc_vec(self, fields:list|None=None):
    if self.mesh is None:
      self.create_mesh()
    self.write_coords_petsc_vec(fname=os.path.join(self.root,"data_coordinates.pvec"))
    step = 0
    for time_entry in self.xdmfdata.timeseries:
      time = float(time_entry)
      step_prefix = str(step).zfill(6)
      print(f"Processing step: {step}, time: {time_entry}")
      self.add_fields_to_mesh(fields=fields, step=step)
      self.mesh = self.mesh.point_data_to_cell_data()
      self.write_step_petsc_vec(fields=fields, step=step)
      step += 1
    return
