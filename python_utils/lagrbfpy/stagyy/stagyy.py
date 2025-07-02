import os
import numpy as np
import pyvista as pvs
import h5py
from .timeseries_filters import XDMFDataFilter

class StagYY:
  def __init__(self, root:str='.'):
    self.root = root
    return
  
class StagYYh5(StagYY):
  def __init__(self,xdmfile:str, root:str='.'):
    super().__init__(root)
    try:
      self.xdmfdata = XDMFDataFilter(xdmfile)
    except Exception as e:
      raise ValueError(f"Failed to process XDMF file {xdmfile}: {e}")
    self.mesh = None

  def mesh_from_file(self, fname:str) -> pvs.StructuredGrid:
    if self.mesh is not None:
      return self.mesh
    with h5py.File(fname, 'r') as fp: # read hdf5 file
      entries = list(fp.keys())  # get all entries in the file
      shape = fp[entries[0]].shape
      coor  = np.zeros((shape[0]*shape[1]*shape[2], 3), dtype=np.float64)
      if shape[0] == 1: # 2D mesh
        permutation = {0:1, 1:2, 2:0} # Y-Z plane, X is 0
      else: # 3D mesh
        permutation = {0:0, 1:1, 2:2} # X-Y-Z order
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


def test():
  import os

  root = os.path.join(os.path.join(os.environ['M3D_DIR'], 'Stagyy', 'test_convection'))
  mdir = os.path.join(root, "hdf5")
  xdmfname = os.path.join(mdir, "Data.xmf")
  stag = StagYYh5(xdmfname,root=mdir)
  print(stag.xdmfdata.timeseries)
  print(stag.xdmfdata.fieldnames)
  print(stag.xdmfdata.filenames)
  stag.create_mesh()
  print(stag.mesh)

if __name__ == '__main__':
  test()