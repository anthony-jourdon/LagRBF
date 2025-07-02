import os
import numpy as np
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