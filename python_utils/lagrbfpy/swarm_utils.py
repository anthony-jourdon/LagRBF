import numpy as np
import pyvista as pvs

try:
  from .utils.halton_seq import halton_sequence
except ImportError:
  from lagrbfpy.utils import halton_sequence

class PointSwarm(pvs.PolyData):
  """
  A simple wrapper around pyvista.PolyData to represent a set of points in space.
  """
  def __init__(
      self, 
      *args, 
      **kwargs
    ) -> None:
    """
    Initialize the PointSwarm object.
    """
    self.point_type = kwargs.pop("point_type", None)
    if self.point_type is None:
      super().__init__(*args, **kwargs)
    else:
      O = kwargs.pop("O", None)
      L = kwargs.pop("L", None)
      N = kwargs.pop("n", None)
      if O is None or L is None or N is None:
        raise ValueError("O, L, and n must be provided when point_type is specified.")
      coor_type = kwargs.pop("coordinate_type", "cartesian")

      TYPE = self.point_type + "_" + coor_type
      match TYPE:
        case "structured_cartesian":
          points = self.structured_coordinates(O, L, N)
        case "structured_spherical":
          points = self.spherical_coordinates(
            self.structured_coordinates(O, L, N), dim=O.shape[0]
          ) 
        case "structured_torus":
          points = self.torus_coordinates(
            self.structured_coordinates(O, L, N), R=kwargs.pop("R", 1.0)
          )
        case "halton_cartesian":
          points = self.halton_coordinates(O, L, N)
        case "halton_spherical":
          points = self.halton_spherical_coordinates(O, L, N)
        case "halton_torus":
          points = self.halton_torus_coordinates(O, L, N, R=kwargs.pop("R", 1.0))
        case _:
          raise ValueError(f"Unknown point_type and coordinate_type combination: {TYPE}. Supported combinations are 'structured_cartesian', 'structured_spherical', 'halton_cartesian', and 'halton_spherical'.")
      super().__init__(points)

  def structured_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:np.ndarray|list|tuple) -> np.ndarray:
    if isinstance(O, (list, tuple)):
      O = np.asarray(O, dtype=np.float64)
    if isinstance(L, (list, tuple)):
      L = np.asarray(L, dtype=np.float64)
    if isinstance(n, (list, tuple)):
      n = np.asarray(n, dtype=np.int32)
    # get spatial dimension
    dim = O.shape[0]
    if dim < 2 or dim > 3:
      raise ValueError(f"Dimension must be 2, or 3. Found dim = {dim}")
    
    x = 3*[None]
    for d in range(dim):
      if n[d] < 1:
        raise ValueError(f"Number of points along dimension {d} must be at least 1. Found n[{d}] = {n[d]}")
      x[d] = np.linspace(O[d], L[d], n[d])
    
    X = np.meshgrid(*x, indexing="ij")
    points = np.zeros((np.prod(n), 3), dtype=np.float64)
    for d in range(dim):
      points[:, d] = X[d].ravel(order="F")
    return points

  def halton_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:int|np.integer) -> np.ndarray:
    if isinstance(O, (list, tuple)):
      O = np.asarray(O, dtype=np.float64)
    if isinstance(L, (list, tuple)):
      L = np.asarray(L, dtype=np.float64)
    if isinstance(n, (list, tuple, np.ndarray)):
      npoints = np.asarray(n, dtype=np.int64).prod()
    else:
      npoints = int(n)

    dim = O.shape[0]
    if dim < 2 or dim > 3:
      raise ValueError(f"Dimension must be 2, or 3. Found dim = {dim}")
    if npoints < 1:
      raise ValueError(f"Number of points must be at least 1. Found n = {npoints}")
    # Generate Halton sequence points in [0,1]^dim
    halton_points = np.zeros((npoints, dim), dtype=np.float64)
    bases = [2, 3, 5]  # Use first three prime numbers as bases for 3D
    for d in range(dim):
      base = bases[d]  # Use different bases for each dimension
      indices = np.arange(npoints, dtype=np.int64)
      halton_points[:, d] = halton_sequence(indices+1, base)
    # Scale and translate to the desired domain [O, L]
    points = np.zeros((npoints, 3), dtype=np.float64)
    points[:, :dim] = O + halton_points * (L - O)
    return points

  def spherical_coordinates(self, points_r:np.ndarray, dim:int) -> np.ndarray:
    points_x = np.zeros_like(points_r)
    if dim == 2:
      points_x[:, 0] = points_r[:, 0] * np.cos(points_r[:, 1])
      points_x[:, 1] = points_r[:, 0] * np.sin(points_r[:, 1])
      points_x[:, 2] = 0.0
    elif dim == 3:
      points_x[:, 0] = points_r[:, 0] * np.sin(points_r[:, 1]) * np.cos(points_r[:, 2])
      points_x[:, 1] = points_r[:, 0] * np.sin(points_r[:, 1]) * np.sin(points_r[:, 2])
      points_x[:, 2] = points_r[:, 0] * np.cos(points_r[:, 1])
    else:
      raise ValueError(f"Dimension must be 2 or 3. Found dim = {dim}")
    return points_x
  
  def halton_spherical_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:int|np.integer) -> np.ndarray:
    if isinstance(O, (list, tuple)):
      O = np.asarray(O, dtype=np.float64)
    if isinstance(L, (list, tuple)):
      L = np.asarray(L, dtype=np.float64)
    if isinstance(n, (list, tuple, np.ndarray)):
      npoints = np.asarray(n, dtype=np.int64).prod()
    else:
      npoints = int(n)

    dim = O.shape[0]
    if dim < 2 or dim > 3:
      raise ValueError(f"Dimension must be 2, or 3. Found dim = {dim}")
    
    indices = np.arange(npoints, dtype=np.int64)
    params  = np.zeros((npoints, 3), dtype=np.float64)
    bases = [2, 3, 5]  # Use first three prime numbers

    if dim == 2:
      u     = halton_sequence(indices, bases[0])
      params[:,0] = np.sqrt( ( L[0]*L[0] - O[0]*O[0] )*u + O[0]*O[0] )
      params[:,1] = O[1] + indices * (L[1] - O[1]) / (npoints - 1)
    elif dim == 3:
      u = halton_sequence(indices+1, bases[0])
      v = halton_sequence(indices+1, bases[1])
      w = halton_sequence(indices+1, bases[2])
      params[:,0] = np.power((L[0]*L[0]*L[0] - O[0]*O[0]*O[0])*u + O[0]*O[0]*O[0], 1.0/3.0) 
      params[:,1] = np.arccos(1.0 - 2.0*v)
      params[:,2] = O[2] + w * (L[2] - O[2])

    points = self.spherical_coordinates(params, dim)

    return points
  
  def torus_coordinates(self, points_r:np.ndarray, R:float) -> np.ndarray:
    points_x = np.zeros_like(points_r)
    points_x[:, 0] = (R + points_r[:, 0] * np.sin(points_r[:, 1])) * np.cos(points_r[:, 2])
    points_x[:, 1] = (R + points_r[:, 0] * np.sin(points_r[:, 1])) * np.sin(points_r[:, 2])
    points_x[:, 2] = points_r[:, 0] * np.cos(points_r[:, 1])
    return points_x
  
  def halton_torus_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:int|np.integer, R:float) -> np.ndarray:
    if isinstance(O, (list, tuple)):
      O = np.asarray(O, dtype=np.float64)
    if isinstance(L, (list, tuple)):
      L = np.asarray(L, dtype=np.float64)
    if isinstance(n, (list, tuple, np.ndarray)):
      npoints = np.asarray(n, dtype=np.int64).prod()
    else:
      npoints = int(n)

    dim = O.shape[0]
    if dim != 3:
      raise ValueError(f"Dimension must be 3 for torus coordinates. Found dim = {dim}")
    
    indices = np.arange(npoints, dtype=np.int64)
    params  = np.zeros((npoints, 3), dtype=np.float64)
    bases = [2, 3, 5]  # Use first three prime numbers

    u = halton_sequence(indices+1, bases[0])
    v = halton_sequence(indices+1, bases[1])
    w = halton_sequence(indices+1, bases[2])
    
    params[:,0] = np.sqrt( ( L[0]*L[0] - O[0]*O[0] )*u + O[0]*O[0] )
    params[:,1] = np.arccos(1.0 - 2.0*v)
    params[:,2] = O[2] + w * (L[2] - O[2])

    points = self.torus_coordinates(params, R)

    return points

def test():
  #O = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
  #L = np.array([1.0, 1.0, 1.0], dtype=np.float64)
  O = np.array([2.185, 0.0, 0.0], dtype=np.float64)
  L = np.array([2.19, np.pi, 2.0*np.pi], dtype=np.float64)
  n = np.array([2, 45, 45], dtype=np.int32)
  #n = 2048

  swarm = PointSwarm(point_type="halton", coordinate_type="spherical", O=O, L=L, n=n)
  swarm.point_data["pidx"] = np.arange(swarm.n_points, dtype=np.int32)

  plotter = pvs.Plotter()
  plotter.add_mesh(swarm, scalars="pidx", point_size=10, render_points_as_spheres=True)
  plotter.show_grid()
  plotter.show()

  return

if __name__ == "__main__":
  test()