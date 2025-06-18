import os
import numpy as np
import petscvec_loader as load
import pyvista as pvs

def write_vtp(coords:np.ndarray, point_data:dict, filename:str):
  """
  Write a VTP file with the given coordinates and point data.
  """
  # ensure coords is a 2D array
  if coords.ndim != 2:
    raise ValueError(f"Expected coords to be a 2D array, got {coords.ndim}D array with shape {coords.shape}")
  dim = coords.shape[1]
  # vtk files need 3D coordinates, so if the input is 2D, we add a third dimension with zeros
  if dim == 2:
    coords = np.concatenate((coords, np.zeros((coords.shape[0], 1))), axis=1)  # Add z-coordinate

  pvdata = pvs.PolyData(coords.astype(np.float64))  # Ensure coordinates are in float64 format
  for key, value in point_data.items():
    pvdata.point_data[key] = value
  # Save the PolyData to a VTP file
  if not filename.endswith('.vtp'):
    filename += '.vtp'
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
  pvdata.save(filename)
  return

def test():
  output_dir = os.path.join(os.path.dirname(__file__), '../output')

  data_fnames = {
    'coords': os.path.join(output_dir, 'swarm_data_coords.pbvec'),
    'fields': os.path.join(output_dir, 'swarm_data_fields.pbvec'),
  }
  interp_fnames = {
    'coords': os.path.join(output_dir, 'swarm_interp_coords.pbvec'),
    'fields': os.path.join(output_dir, 'swarm_interp_fields.pbvec'),
  }

  dim          = load.PetscVecLoadBlockSizeFromFile(data_fnames['coords'] + '.info')
  nfields      = load.PetscVecLoadBlockSizeFromFile(data_fnames['fields'] + '.info')
  data_coor    = load.PetscVecLoadFromFile(data_fnames['coords'])
  data_field   = load.PetscVecLoadFromFile(data_fnames['fields'])
  interp_coor  = load.PetscVecLoadFromFile(interp_fnames['coords'])
  interp_field = load.PetscVecLoadFromFile(interp_fnames['fields'])

  data_coor    = np.reshape(data_coor, (-1, dim))
  data_field   = np.reshape(data_field, (-1, nfields))
  interp_coor  = np.reshape(interp_coor, (-1, dim))
  interp_field = np.reshape(interp_field, (-1, nfields))

  npoints = data_coor.shape[0]
  if dim == 2:
    data_coor = np.concatenate((data_coor, np.zeros((npoints, 1))), axis=1)  # Add z-coordinate
    interp_coor = np.concatenate((interp_coor, np.zeros((interp_coor.shape[0], 1))), axis=1)  # Add z-coordinate

  # Create a PyVista point set for the data
  pvdata:pvs.PolyData = pvs.PolyData(data_coor)
  for i in range(nfields):
    pvdata.point_data[f'f{i}'] = data_field[:, i]
  pvdata.save(os.path.join(os.environ['M3D_DIR'], 'Stagyy', 'output', 'swarm_data.vtp'))
  # Create a PyVista point set for the interpolation
  pvinterp:pvs.PolyData = pvs.PolyData(interp_coor)
  for i in range(nfields):
    pvinterp.point_data[f'f{i}'] = interp_field[:, i]
  pvinterp.save(os.path.join(os.environ['M3D_DIR'], 'Stagyy', 'output', 'swarm_interp.vtp'))

  return

def test2():
  output_dir = os.path.join(os.path.dirname(__file__), '../output')

  data_fnames = {
    'coords': os.path.join(output_dir, 'swarm_data_coords.pbvec'),
    'fields': os.path.join(output_dir, 'swarm_data_fields.pbvec'),
  }
  interp_fnames = {
    'coords': os.path.join(output_dir, 'swarm_interp_coords.pbvec'),
    'fields': os.path.join(output_dir, 'swarm_interp_fields.pbvec'),
  }
  data_coor    = load.PetscVecLoad(data_fnames['coords'])
  data_field   = load.PetscVecLoad(data_fnames['fields'])
  data = {
    'f0': data_field[:, 0],
    'f1': data_field[:, 1],
    'f2': data_field[:, 2],
  }
  interp_coor  = load.PetscVecLoad(interp_fnames['coords'])
  interp_field = load.PetscVecLoad(interp_fnames['fields'])
  interp = {
    'f0': interp_field[:, 0],
    'f1': interp_field[:, 1],
    'f2': interp_field[:, 2],
  }
  write_vtp(data_coor, data, os.path.join(os.environ['M3D_DIR'], 'Stagyy', 'output', 'swarm_data.vtp'))
  write_vtp(interp_coor, interp, os.path.join(os.environ['M3D_DIR'], 'Stagyy', 'output', 'swarm_interp.vtp'))
  print("VTP files created successfully.")
  

  return

if __name__ == '__main__':
  #test()
  #print("VTP files created successfully.")
  test2()