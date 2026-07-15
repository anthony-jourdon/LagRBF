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