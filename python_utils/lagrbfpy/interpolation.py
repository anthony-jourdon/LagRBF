import os
import numpy as np
import lagrbfpy as lrb
from petsc4py import PETSc

class LagRBFInterpolator:
  """
  Radial-basis-function interpolator backed by the parallel C/PETSc library.

  Parallel usage
  --------------
  Launch with MPI and let petsc4py initialise PETSc/MPI before any PETSc
  call is made::

      mpiexec -n 4 python my_script.py

  All MPI ranks must create the interpolator and call ``interpolate()``
  collectively (the underlying C code is a collective PETSc operation).
  
  Each rank can provide its own subset of data points and query points;
  ``DMSwarmAssignMigrateBasic()`` handles routing to the correct rank for
  computation.
  
  The returned array contains valid results **only on rank 0**; other ranks
  receive a zero-filled array of the correct shape. Use ``PETSc.COMM_WORLD.bcast``
  if you need the results on every rank.
  """

  def __init__(self, data_coords:np.ndarray|None=None, data_values:np.ndarray|None=None, petsc_options:dict|None=None, **kwargs) -> None:
    """
    Initialize the interpolator.

    Parameters
    ----------
    data_coords : (n, dim) ndarray or None
        Coordinates of data points on this rank. If None, assume rank 0 has no data.
    data_values : (n,) or (n, nfields) ndarray or None
        Field values at data points on this rank. If None, assume rank 0 has no data.
    petsc_options : dict, optional
        PETSc options. If None, call PETSc.Options() constructor.
    **kwargs : dict
        Configuration:
        - partitions: (dim,) int32 array, default [8, 8, ..., 8]
        - O: domain origin, default auto-computed from data_coords
        - L: domain end, default auto-computed from data_coords
        - padding_factor: float, default 0.1
        - overlap: float, default 1.0
        - shape_fac: float, default 1.0
        - kernel: int, default 2 (Wendland C2)
        - log_view: bool, default False
    """
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    self.real_dtype = np.dtype(PETSc.RealType)
    self.int_dtype = np.dtype(PETSc.IntType)
    self.options_file = None
    
    # Handle None data (rank may have no points)
    if data_coords is None:
      data_coords = np.empty((0, 1), dtype=self.real_dtype)
    if data_values is None:
      data_values = np.empty((0, 1), dtype=self.real_dtype)
    
    if data_coords.ndim == 1 and data_coords.shape[0] > 0:
      raise ValueError(f"data_coords should be 2D (n, dim), found shape {data_coords.shape}")
    
    self.data_coords = data_coords
    self.data_values = data_values
    
    # Infer dimension and number of fields
    if self.data_coords.shape[0] > 0:
      self.dim = self.data_coords.shape[1]
    else:
      self.dim = kwargs.get("dim", 3)  # default to 3D if no data

    options = petsc_options
    if options is None:
      options:dict = PETSc.Options()

    options["pu_mesh_dim"] = self.dim
    
    if self.data_values.ndim > 1 and self.data_values.shape[0] > 0:
      self.nfields = self.data_values.shape[1]
    else:
      self.nfields = kwargs.get("nfields", 1)
    options["n_interpolation_fields"] = self.nfields
    
    # Ensure data_values has shape (n, nfields)
    if self.data_values.size == 0:
      self.data_values = np.empty((0, self.nfields), dtype=self.real_dtype)
    elif self.data_values.ndim == 1:
      self.data_values = self.data_values.reshape(-1, 1)
    
    options["pu_mesh_O"] = kwargs.get("O", self.bbox_origin(kwargs.get("padding_factor", 0.1)))
    options["pu_mesh_L"] = kwargs.get("L", self.bbox_end(kwargs.get("padding_factor", 0.1)))

    part = np.full(self.dim, 8, dtype=np.int32)
    options["pu_mesh_n"]            = kwargs.get("partitions", part)
    options["pu_mesh_overlap"]      = kwargs.get("overlap", 1.0)
    options["pu_mesh_shape_factor"] = kwargs.get("shape_fac", 1.0)

    #options["log_view"]     = kwargs.get("log_view", False)
    options["kernel_type"]  = kwargs.get("kernel", 2)
    options["solve_method"] = 2
    options["partition_pc_type"] = "cholesky"
    options["partition_ksp_type"] = "preonly"
    options["partition_ksp_reuse_preconditioner"] = True
    options["partition_pc_factor_shift_type"] = "positive_definite"
    
    return
  
  def bbox_origin(self, padding_factor:float=0.1) -> np.ndarray:
    if self.data_coords.shape[0] == 0:
      return np.zeros(self.dim, dtype=self.real_dtype)
    min_coords = np.min(self.data_coords, axis=0)
    min_dx = padding_factor * np.abs(min_coords)
    O = min_coords - min_dx
    return O
  
  def bbox_end(self, padding_factor:float=0.1) -> np.ndarray:
    if self.data_coords.shape[0] == 0:
      return np.ones(self.dim, dtype=self.real_dtype)
    max_coords = np.max(self.data_coords, axis=0)
    max_dx = padding_factor * np.abs(max_coords)
    L = max_coords + max_dx
    return L
  
  def interpolate(self, points_coords:np.ndarray|None=None, npoints_global:int|None=None) -> np.ndarray:
    """
    Interpolate at query points.

    Must be called collectively by all MPI ranks.
    
    Parameters
    ----------
    points_coords : (n, dim) ndarray or None
        Query point coordinates on this rank. If None, this rank contributes no points.
    npoints_global : int, optional
        Total number of query points across all ranks. If None, computed via MPI AllReduce.
    
    Returns
    -------
    results : (npoints_global, nfields) ndarray
        Interpolated values. Valid on all ranks.
    """
    comm = PETSc.COMM_WORLD
    
    if points_coords is None:
      points_coords = np.empty((0, self.dim), dtype=self.real_dtype)
    
    npoints_local = self.int_dtype.type(points_coords.shape[0])
    
    # Compute global total if not provided
    if npoints_global is None:
      npts_local_py  = int(npoints_local)
      npts_global_py = comm.allreduce(npts_local_py)
      npoints_global = npts_global_py
    
    # Prepare arrays with PETSc real precision and C-contiguous layout
    data_coords = self.data_coords.astype(self.real_dtype, copy=True)
    data_coords_flat = data_coords.ravel()
    data_values = self.data_values.astype(self.real_dtype, copy=True)
    data_values_flat = data_values.ravel()
    
    points_coords_arr  = points_coords.astype(self.real_dtype, copy=True)
    points_coords_flat = points_coords_arr.ravel()
    
    # Allocate output: full size on rank 0, local size on others for MPI gather
    points_values = np.zeros(npoints_global * self.nfields, dtype=self.real_dtype)

    lrb.cfuncs["lagrbf_interpolate"](
      self.int_dtype.type(self.data_coords.shape[0]),
      data_coords_flat,
      data_values_flat,
      npoints_local,
      points_coords_flat,
      points_values,
      self.int_dtype.type(npoints_global),
      self.options_file if self.options_file is None else self.options_file.encode("utf-8")
    )
    
    points_values = points_values.reshape((npoints_global, self.nfields))
    return points_values