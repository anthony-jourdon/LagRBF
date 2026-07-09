from dataclasses import dataclass
import numpy as np

@dataclass
class PointContainer:
  """
  A simple container for points and their associated data.
  """
  coords: np.ndarray
  values: np.ndarray|None = None
  pidx: np.ndarray|None = None