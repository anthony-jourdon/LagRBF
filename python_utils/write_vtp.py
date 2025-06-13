import os
import numpy as np
import petscvec_loader as load
import pyvista as pvs

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

if __name__ == '__main__':
  test()
  print("VTP files created successfully.")