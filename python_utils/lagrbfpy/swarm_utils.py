import numpy as np
import pyvista as pvs
from petsc4py import PETSc

try:
  from .utils.halton_seq import halton_sequence
except ImportError:
  from lagrbfpy.utils import halton_sequence

class PointSwarm(pvs.PolyData):
  """
  A simple wrapper around PyVista's :py:class:`PolyData <pyvista.PolyData>` to represent a set of points in space.
  The constructor can take any PyVista :py:class:`PolyData <pyvista.PolyData>` constructor arguments, 
  but also allows for the generation of structured or Halton point distributions in Cartesian, spherical, 
  or toroidal domain.

  :param str point_type:
    The type of point distribution to generate. 
    Can be ``"structured"`` or ``"halton"``. 
    If None, the points are taken from the provided :py:class:`PolyData <pyvista.PolyData>` arguments.
  :param str coordinate_type:
    The type of coordinates to use.
    Can be ``"cartesian"``, ``"spherical"``, or ``"toroidal"``. 
    If None, defaults to ``"cartesian"``.
  :param numpy.ndarray O:
    The minimum coordinates (lower bounds) of the point distribution.
  :param numpy.ndarray L:
    The maximum coordinates (upper bounds) of the point distribution.
  :param numpy.ndarray n:
    The number of points. 
    If ``point_type`` is ``"structured"``, this should be an array of integers specifying 
    the number of points along each dimension.
    If ``point_type`` is ``"halton"``, this can be a single integer specifying 
    the total number of points, if an array of integers is provided, the product of its entries is done.
  :param float R:
    The major radius of the torus, used only if ``coordinate_type`` is ``"toroidal"``.
  :param petsc4py.PETSc.Comm comm:
    The MPI communicator to use for parallel point generation.
    If not provided, defaults to ``PETSc.COMM_SELF`` (no parallelism).
  :param list bases:
    The bases to use for the Halton sequence generation.
    If not provided, defaults to the first three prime numbers (2, 3, 5) for the 
    x, y, and z dimensions respectively. **Must be prime numbers**.
  """
  def __init__(
      self, 
      *args, 
      **kwargs
    ) -> None:
    self.point_type = kwargs.pop("point_type", None)
    if self.point_type is None:
      super().__init__(*args, **kwargs)
    else:
      comm = kwargs.pop("comm", None)
      if comm is not None:
        self.comm = comm
        self.rank = comm.getRank()
        self.size = comm.getSize()
      else:
        self.comm = PETSc.COMM_SELF
        self.rank = 0
        self.size = 1

      O = kwargs.pop("O", None)
      L = kwargs.pop("L", None)
      N = kwargs.pop("n", None)
      if O is None or L is None or N is None:
        raise ValueError("O, L, and n must be provided when point_type is specified.")
      coor_type = kwargs.pop("coordinate_type", "cartesian")

      # Determine the number of points per rank for Halton sequence generation
      self.bases = None
      if self.point_type == "halton":
        N = self.halton_points_repart(N)
        self.bases = kwargs.pop("bases", [2, 3, 5])  # Default bases for Halton sequence

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

  def halton_points_repart(self, N:int|np.integer|list|tuple|np.ndarray) -> int:
    """
    Calculate the number of points to be generated on each rank for a Halton sequence,
    given the total number of points ``N`` and the number of MPI ranks.
    In addition, it stores the total number of points across all ranks (:py:attr:`global_npoints`) and 
    the number of points per rank in the instance attributes (:py:attr:`npoints_per_rank`).

    :param int|numpy.integer|list|tuple|numpy.ndarray N:
      The total number of points to be generated. 
      If a list, tuple, or numpy array is provided, 
      the product of its entries is taken as the total number of points.
    :return:
      The number of points to be generated on the current rank.
    :rtype:
      int
    """
    if isinstance(N, (list, tuple, np.ndarray)):
      N = np.asarray(N, dtype=np.int64).prod()
    else:
      N = int(N)
    # total number of points across all ranks
    self.global_npoints   = N
    # number of points per rank
    self.npoints_per_rank = self.global_npoints // self.size
    # remainding points to be distributed to the last rank (if the integer division is not exact)
    npoints_remainder = self.global_npoints % self.size
    # rank local number of points
    if self.rank == self.size - 1:
      local_npoints = self.npoints_per_rank + npoints_remainder
    else:
      local_npoints = self.npoints_per_rank
    return local_npoints

  def structured_coordinates_SEQ(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:np.ndarray|list|tuple) -> np.ndarray:
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

  def structured_coordinates_MPI(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:np.ndarray|list|tuple) -> np.ndarray:
    # get spatial dimension
    dim = O.shape[0]
    if dim < 2 or dim > 3:
      raise ValueError(f"Dimension must be 2, or 3. Found dim = {dim}")
    
    # use petsc DMDA to generate structured grid of points in parallel
    da = PETSc.DMDA().create(dim=dim, dof=1, sizes=n, comm=self.comm)
    da.setUniformCoordinates(O[0], L[0], O[1], L[1], O[2], L[2])
    coords    = da.getCoordinates()
    bs        = coords.getBlockSize()
    LA_coords = coords.getArray(readonly=True).reshape(-1, bs)
    da.destroy()
    return LA_coords

  def structured_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:np.ndarray|list|tuple) -> np.ndarray:
    """
    Generate an uniform structured grid of points.
    The points are generated in a grid defined by the lower bounds ``O``, upper bounds ``L``, 
    and the number of points ``n`` along each dimension.

    :param numpy.ndarray|list|tuple O:
      The minimum coordinates (lower bounds) of the grid.
    :param numpy.ndarray|list|tuple L:
      The maximum coordinates (upper bounds) of the grid.
    :param numpy.ndarray|list|tuple n:
      The number of points along each dimension. Should be an array of integers.
    :return:
      An array of shape ``(N, 3)`` containing the generated points, where ``N`` 
      is the total number of points (product of entries in `n`).
    :rtype: 
      numpy.ndarray

    .. note::
      In parallel, the points are generated using PETSc's DMDA (Distributed Array) functionality,
      which distributes the points across the available MPI ranks.

    """
    if isinstance(O, (list, tuple)):
      O = np.asarray(O, dtype=np.float64)
    if isinstance(L, (list, tuple)):
      L = np.asarray(L, dtype=np.float64)
    if isinstance(n, (list, tuple)):
      n = np.asarray(n, dtype=np.int32)
    if self.size > 1:
      points = self.structured_coordinates_MPI(O, L, n)
    else:
      points = self.structured_coordinates_SEQ(O, L, n)
    return points
  
  def halton_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:int|np.integer) -> np.ndarray:
    """
    Generate a set of points in Cartesian coordinates using the 
    :py:func:`Halton sequence <lagrbfpy.halton_sequence>`.
    The points are generated in a box defined by the lower bounds ``O``, upper bounds ``L``, 
    and the total number of points ``n``.

    .. note::
      The bases used for the Halton sequence are the first three prime numbers (2, 3, 5) 
      for the x, y, and z dimensions respectively.

    :param numpy.ndarray|list|tuple O:
      The minimum coordinates (lower bounds) of the box.
    :param numpy.ndarray|list|tuple L:
      The maximum coordinates (upper bounds) of the box.
    :param int n:
      The total number of points to generate.
    :return:
      An array of shape ``(n, 3)`` containing the generated points.
    :rtype:
      numpy.ndarray
    """
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
    bases = self.bases
    indices = np.arange(npoints, dtype=np.int64) + self.rank*self.npoints_per_rank
    for d in range(dim):
      base = bases[d]  # Use different bases for each dimension
      halton_points[:, d] = halton_sequence(indices+1, base)
    # Scale and translate to the desired domain [O, L]
    points = np.zeros((npoints, 3), dtype=np.float64)
    points[:, :dim] = O + halton_points * (L - O)
    return points

  def spherical_coordinates(self, points_r:np.ndarray, dim:int) -> np.ndarray:
    """
    Generate a structured uniform Cartesian coordinates sphere from spherical coordinates.

    .. note::
      The point distribution is called "uniform" because the points are generated in a structured manner,
      but the actual distribution of points on the sphere **is not** uniform, the distance between points
      varies with the polar angle.
      For a truly uniform distribution of points on a sphere, consider using the 
      :py:meth:`halton_spherical_coordinates` method.

    :param numpy.ndarray points_r:
      An array of shape ``(N, 3)`` containing the spherical coordinates of the points,
      where ``N`` is the total number of points. 
      The first column should contain the radial distances, 
      the second column should contain the polar angles (in radians), 
      and the third column should contain the azimuthal angles (in radians).
      This parameter can be generated using the :py:meth:`structured_coordinates` method.
    :param int dim:
      The spatial dimension of the points. Should be either 2 or 3.
    :return:
      An array of shape ``(N, 3)`` containing the generated Cartesian coordinates of the points.
    :rtype:
      numpy.ndarray
    """
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
    """
    Generate a set of points in a sphere (or a disc) using a :py:func:`Halton sequence <lagrbfpy.halton_sequence>`.
    
    .. note::
      The bases used for the Halton sequence are the first three prime numbers (2, 3, 5) 
      for the radial, polar, and azimuthal dimensions respectively.
      This method generate a point distribution that fill the spherical domain much more uniformly 
      than the :py:meth:`structured_coordinates` method.

    :param numpy.ndarray|list|tuple O:
      The minimum coordinates (lower bounds) of the spherical domain.
    :param numpy.ndarray|list|tuple L:
      The maximum coordinates (upper bounds) of the spherical domain.
    :param int n:
      The total number of points to generate.
    :return:
      An array of shape ``(N, 3)`` containing the generated points, where ``N`` 
      is the total number of points (product of entries in ``n``).
    :rtype:
      numpy.ndarray
    """
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
    
    indices = np.arange(npoints, dtype=np.int64) + self.rank*self.npoints_per_rank
    params  = np.zeros((npoints, 3), dtype=np.float64)
    bases = self.bases

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
    """
    Convert torus coordinates to Cartesian coordinates.
    
    :param numpy.ndarray points_r:
      The torus coordinates (radial, polar, azimuthal).
    :param float R:
      The major radius of the torus.
    :return:
      An array of shape ``(N, 3)`` containing the Cartesian coordinates, where ``N`` 
      is the total number of points.
    :rtype:
      numpy.ndarray
    """
    points_x = np.zeros_like(points_r)
    points_x[:, 0] = (R + points_r[:, 0] * np.sin(points_r[:, 1])) * np.cos(points_r[:, 2])
    points_x[:, 1] = (R + points_r[:, 0] * np.sin(points_r[:, 1])) * np.sin(points_r[:, 2])
    points_x[:, 2] = points_r[:, 0] * np.cos(points_r[:, 1])
    return points_x
  
  def halton_torus_coordinates(self, O:np.ndarray|list|tuple, L:np.ndarray|list|tuple, n:int|np.integer, R:float) -> np.ndarray:
    """
    Generate a set of points in a torus using a :py:func:`Halton sequence <lagrbfpy.halton_sequence>`.
    The points are generated in a toroidal domain defined by the lower bounds ``O``, upper bounds ``L``, 
    and the total number of points ``n``.
    The major radius of the torus is specified by ``R``.

    :param numpy.ndarray|list|tuple O:
      The minimum coordinates (lower bounds) of the toroidal domain.
    :param numpy.ndarray|list|tuple L:
      The maximum coordinates (upper bounds) of the toroidal domain.
    :param int n:
      The total number of points to generate.
    :param float R:
      The major radius of the torus.
    :return:
      An array of shape ``(N, 3)`` containing the generated points, 
      where ``N`` is the total number of points (product of entries in ``n``).
    :rtype:
      numpy.ndarray
    """
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
    
    indices = np.arange(npoints, dtype=np.int64) + self.rank*self.npoints_per_rank
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