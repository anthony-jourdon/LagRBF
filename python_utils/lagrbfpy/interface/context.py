import os
import numpy as np
import ctypes
from petsc4py import PETSc
from pyvista import PolyData

from .. import cfuncs, print
from ..utils import PointContainer

class LagRBFCtx:
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    """
    Options logic:
    Options provided by keyword arguments are priorised, then if user provided already 
    existing options dictionnary we look into it and finally if nothing is found we attribute a 
    default value.
    """
    self.data_filled:bool   = False
    self.interp_filled:bool = False
    self.PetscReal:np.dtype = np.dtype(PETSc.RealType)
    self.PetscRealCtype     = np.ctypeslib.as_ctypes_type(self.PetscReal)

    self.options = petsc_options
    if self.options is None:
      self.options:dict = PETSc.Options()

    self.output_path = kwargs.get("output_path", self.options.getString("output_path", "output"))
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)
      print(f"Created output directory: {self.output_path}")

    self.dim = kwargs.get("dim", self.options.getInt("pu_mesh_dim", 3) )
    # spatial dimension of the problem
    self.options["pu_mesh_dim"] = self.dim
    # Partition of Unity bounding box
    self.options["pu_mesh_O"] = kwargs.get(
      "minCoords", 
      self.options.getRealArray(
        "pu_mesh_O", 
        np.zeros(self.dim, dtype=self.PetscReal)
      )
    )
    self.options["pu_mesh_L"] = kwargs.get(
      "maxCoords", 
      self.options.getRealArray(
        "pu_mesh_L", 
        np.ones(self.dim, dtype=self.PetscReal)
      )
    )
    # Number of partitions
    self.options["pu_mesh_n"] = kwargs.get(
      "npartitions",
      self.options.getIntArray(
        "pu_mesh_n", 
        np.full(self.dim, 4, dtype=np.int32)          
      )
    )
    # Number of fields to interpolate, can be 0 if only advection from analytical function is required
    self.options["n_interpolation_fields"] = kwargs.get(
      "nfields", self.options.getInt("n_interpolation_fields", 0)
    )
    self.options["pu_mesh_overlap"] = kwargs.get(
      "overlap", self.options.getReal("pu_mesh_overlap", 1.0)
    )
    self.options["pu_mesh_shape_factor"] = kwargs.get(
      "shape_factor", self.options.getReal("pu_mesh_shape_factor", 1.0)
    )
    self.options["kernel_type"]  = kwargs.get(
      "kernel", self.options.getInt("kernel_type", 2)
    )
    self.options["solve_method"] = kwargs.get(
      "solve_method", self.options.getInt("solve_method", 2)
    )
    self.options["partition_pc_type"] = kwargs.get(
      "partition_pc_type", self.options.getString("partition_pc_type", "cholesky")
    )
    self.options["partition_ksp_type"] = kwargs.get(
      "partition_ksp_type", self.options.getString("partition_ksp_type", "preonly")
    )
    self.options["partition_ksp_reuse_preconditioner"] = kwargs.get(
      "partition_ksp_reuse_preconditioner", 
      self.options.getBool("partition_ksp_reuse_preconditioner", True)
    )
    self.options["partition_pc_factor_shift_type"] = kwargs.get(
      "partition_pc_factor_shift_type", 
      self.options.getString("partition_pc_factor_shift_type", "positive_definite")
    )
    self.options["advection"] = kwargs.get(
      "advection", self.options.getBool("advection", False)
    )
    self.options["advection_scheme"] = kwargs.get(
      "advection_scheme", self.options.getInt("advection_scheme", 2)
    )
    self.options["cfl_fac"] = kwargs.get(
      "advection_cfl", self.options.getReal("cfl_fac", 0.3)
    )
    self.options["dt"] = kwargs.get(
      "dt", self.options.getReal("dt", 1e-4)
    )
    self.options["rmin"] = kwargs.get(
      "rmin", self.options.getReal("rmin", 0.0)
    )
    self.options["rmax"] = kwargs.get(
      "rmax", self.options.getReal("rmax", 1e132)
    )

    self.ctx:ctypes.c_void_p = cfuncs["lagrbf_create"]()
    if self.ctx is None:
      raise RuntimeError("Failed to create LagRBF context, see PETSc error for details.")
  
    self.refs:dict = {}  # Store references to Python objects to prevent garbage collection

  def get_ctx(self) -> ctypes.c_void_p:
    if self.ctx is None:
      raise RuntimeError("LagRBF context has not been created or has been destroyed.")
    return self.ctx
  
  def get_stored_refs(self) -> dict:
    return self.refs
  
  def get_ref(self, key:str):
    ref = self.refs.get(key, None)
    if ref is None:
      raise KeyError(f"No reference found for key '{key}'.")
    return ref

  def store_ref(self, **objs):
    self.refs.update(objs)

  # lets it be passed directly as c_void_p
  def __index__(self) -> ctypes.c_void_p:
    return self.ctx
  
  def __bool__(self) -> bool:
    return self.ctx is not None
  
  def __del__(self) -> None:
    """
    Call the C function to clear the structure in memory
    """
    if self.ctx is not None:
      cfuncs["lagrbf_destroy"](self.ctx)
      self.ctx = None

  def _begin_c(self, array:np.ndarray, PetscType:np.dtype) -> np.ndarray:
    if array is None:
      serr  = "Input array cannot be None.\n"
      serr += "Consider using an empty array instead: np.empty((0, dim))\n"
      raise ValueError(serr)
    
    c_arr = array.view(PetscType)
    c_arr.shape = (array.size,)
    return c_arr
  
  def _end_c(self, array:np.ndarray, bs:int=1) -> np.ndarray:
    py_arr = array.reshape((-1, bs), order="C")
    return py_arr
  
  def fill_swarm(
      self, 
      coords:np.ndarray, 
      values:np.ndarray|None=None, 
      pidx:np.ndarray|None=None,
      is_data:bool=False
    ) -> None:
    ctx:ctypes.c_void_p = self.ctx
    dim:int = self.dim

    # ensure all ranks have a valid coords array, even if empty
    if coords is None:
      coords = np.empty((0, dim), dtype=self.PetscReal)
    
    npoints = coords.shape[0]
    pcoords = self._begin_c(coords, self.PetscReal)

    pvals = None
    if values is not None:
      pvals = self._begin_c(values, self.PetscReal)
    
    ppidx = None
    if pidx is not None:
      ppidx = self._begin_c(pidx, np.int64)
    
    ierr = cfuncs["lagrbf_fill_swarm"](
      ctx, 
      is_data, 
      npoints, 
      pcoords, 
      pvals, 
      ppidx
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_fill_swarm failed with error code {ierr}, see PETSc error for details.")
    return
  
  def fill_data(self, coords:np.ndarray, values:np.ndarray, pidx:np.ndarray|None=None) -> None:
    self.fill_swarm(coords=coords, values=values, pidx=pidx, is_data=True)
    self.data_filled = True
    return
  
  def fill_points(self, coords:np.ndarray, values:np.ndarray|None=None, pidx:np.ndarray|None=None) -> None:
    self.fill_swarm(coords=coords, values=values, pidx=pidx, is_data=False)
    self.interp_filled = True
    return
  
  def output(
      self,
      coords:np.ndarray|None=None,
      fvals:np.ndarray|None=None,
      pidx:np.ndarray|None=None,
      fieldname:str="fields",
      is_data:bool=False
    ) -> None:

    npoints = 0
    nfields = 0
    if coords is None and fvals is None and pidx is None:
      raise ValueError("At least one of coords, fvals, or pidx must be provided.")
    if coords is not None:
      npoints = coords.shape[0]
      coords  = self._begin_c(coords, self.PetscReal)
    if fvals is not None:
      npoints = fvals.shape[0]
      nfields = fvals.shape[1]
      fvals   = self._begin_c(fvals, self.PetscReal)
    if pidx is not None:
      npoints = pidx.shape[0]
    
    if npoints == 0:
      raise ValueError("Number of points must be greater than zero.")

    ierr = cfuncs["lagrbf_output"](
      self.ctx, 
      is_data, 
      npoints, 
      coords, 
      fvals, 
      pidx, 
      fieldname.encode('utf-8')
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_output failed with error code {ierr}, see PETSc error for details.")
    
    if coords is not None:
      coords = self._end_c(coords, self.dim)
    if fvals is not None:
      fvals = self._end_c(fvals, bs=nfields)
    return

  def save(
      self, 
      comm:PETSc.Comm, 
      fname:str, 
      fields_size:dict[str, int],
      output_points:PointContainer,
      fieldname:str="fields",
      is_data:bool=False,
    ) -> None:

    self.output(
      coords=output_points.coords,
      fvals=output_points.values,
      pidx=output_points.pidx,
      fieldname=fieldname,
      is_data=is_data
    )
    # write to disk only on rank 0
    rank = comm.getRank()
    if rank == 0:
      swarm = PolyData(output_points.coords)
      idx = 0
      for fieldname, bs in fields_size.items():
        swarm.point_data[fieldname] = output_points.values[:, idx:idx+bs]
        idx += bs
      swarm.point_data["pidx"] = output_points.pidx
      swarm.save(fname)
      del swarm
    return

  def interpolate_c(self) -> None:
    if not self.data_filled:
      raise RuntimeError("Data have not been provided, use fill_data() to provide data points")
    if not self.interp_filled:
      raise RuntimeError("Interpolation swarm has not been provided, use fill_points() to provide the points on which to interpolate")

    ctx = self.ctx

    ierr = cfuncs["lagrbf_interpolation"](ctx)
    if ierr != 0:
      raise RuntimeError(f"lagrbf_interpolation failed with error code {ierr}, see PETSc error for details.")
    return

  def reset_coefficients(self) -> None:
    ierr = cfuncs["lagrbf_reset_coefficients"](self.ctx)
    if ierr != 0:
      raise RuntimeError(f"lagrbf_reset_coefficients failed with error code {ierr}, see PETSc error for details.")
    return
  
  def set_advection_function(self, *args, **kwargs) -> None:
    """
    Set the advection function to be used for advecting points. 
    Two possibilities:

    1. Provide the advection function and optional data directly as arguments:
    2. Get the advection function and optional data from stored references using keys.
      By default, it will look for "advection_fn" and "advection_data" keys in the stored references, if not found
      it will look for keys provided in kwargs as "stored_fn_key" and "stored_data_key".
    """
    if len(args) == 0:
      # check for advection_fn and advection_data in stored refs
      fn   = self.get_ref("advection_fn", None)
      data = self.get_ref("advection_data", None)
      if fn is None:
        key = kwargs.get("stored_fn_key", None)
        if key is not None:
          fn = self.get_ref(key, None)
        dkey = kwargs.get("stored_data_key", None)
        if dkey is not None:
          data = self.get_ref(dkey, None)
      if fn is None:
        serr  = "No advection function provided and no stored reference found.\n"
        serr += "Either provide an advection function as an argument or store it using store_ref(advection_fn=your_function, advection_data=your_data)."
        raise ValueError(serr)
      func = fn
    else:
      func = args[0]
      if not callable(func):
        raise ValueError("Provided advection function is not callable.")
      self.store_ref(advection_fn=func)
      if len(args) > 1:
        data = args[1]
        self.store_ref(advection_data=data)
    if not callable(func):
      raise ValueError("Provided advection function is not callable.")
    self.store_ref(advection_function=func)

    ierr = cfuncs["lagrbf_advection_function_batch"](
      self.ctx,
      func,
      ctypes.cast(ctypes.pointer(data), ctypes.c_void_p)
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_advection_function_batch failed with error code {ierr}, see PETSc error for details.")
    return

  def advect_analytical_c(self, time_start:float, time_end:float) -> None:
    if not self.interp_filled:
      raise RuntimeError("Swarm to advect has not been provided, use fill_swarm() to provide the points to advect")
    ierr = cfuncs["lagrbf_avect_analytical"](
      self.ctx,
      self.PetscRealCtype(time_start), # current time
      self.PetscRealCtype(time_end)    # next time
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_advect failed with error code {ierr}, see PETSc error for details.")
    return

  def advect_interpolate_c( self, time_start:float, time_end:float) -> None:
    if not self.data_filled:
      raise RuntimeError("Data have not been provided, use fill_swarm() to provide data points")
    if not self.interp_filled:
      raise RuntimeError("Swarm to advect has not been provided, use fill_swarm() to provide the points to advect")
    # advect
    ierr = cfuncs["lagrbf_avect_interpolate"](
      self.ctx,
      self.PetscRealCtype(time_start),
      self.PetscRealCtype(time_end)     
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_advect failed with error code {ierr}, see PETSc error for details.")
    return