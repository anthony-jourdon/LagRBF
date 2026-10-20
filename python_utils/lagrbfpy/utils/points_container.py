from dataclasses import dataclass
import numpy as np

@dataclass
class PointContainer:
  """
  A simple container for points and their associated data.

  :param numpy.ndarray coords: 
    The coordinates of the points, as a 2D numpy array of shape (npoints, dim).
  :param numpy.ndarray values:
    The values associated with the points, as a 2D numpy array of shape (npoints, nfields). 
    This can be None.
  :param numpy.ndarray pidx:
    The point indices, as a 1D numpy array of shape (npoints,). 
    This can be None.
  """
  coords: np.ndarray
  values: np.ndarray|None = None
  pidx: np.ndarray|None = None

def fill_datafields(npoints, fields:dict[str, int], point_data:dict[str, np.ndarray], nan_replacement:float=0.0) -> np.ndarray:
  """
  Fills a data array with the values from the point data.
  It replaces any NaN values with the specified nan_replacement value. Default to 0.

  :param int npoints:
    The number of points.
  :param dict[str, int] fields:
    A dictionary where keys are field names and values are the number of components for each field.
  :param dict[str, np.ndarray] point_data:
    A dictionary containing the point data, where keys are field names and values are numpy arrays of shape ``(npoints, ncomponents)``.
  :param float nan_replacement:
    The value to replace NaN values with. Default is 0.0.

  :return numpy.ndarray:
    A 2D numpy array of shape ``(npoints, nfields)`` containing the filled data values, where ``nfields`` is the total number of components across all specified fields.
  """
  nfields = sum(fields.values())
  fvals   = np.zeros((npoints, nfields), dtype=np.float64)
  
  idx = 0
  for f in fields:
    if f in point_data:
      # if nans replace with zeros
      fvals[ :, idx:idx+fields[f] ] = np.nan_to_num(point_data[f].reshape((npoints, fields[f])), nan=nan_replacement)
      idx += fields[f]

  return fvals