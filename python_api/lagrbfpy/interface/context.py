"""@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    context.py

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@"""

import os
import numpy as np
import ctypes
from petsc4py import PETSc
from pyvista import PolyData

from . import cfuncs
from .. import print
from ..utils import PointContainer
from .function_pointer import FunctionData

class LagRBFCtx:
  """
  Class to interface the C context and functions for Lagrangian advection and interpolation using
  Radial Basis Functions (RBFs) in parallel using `PETSc`_.
  This is a low level class and it is very unlikely that users will need to use it directly. 
  Instead, use the classes :py:class:`LagRBFInterpolator <lagrbfpy.LagRBFInterpolator>`, 
  :py:class:`AdvectionAnalytical <lagrbfpy.AdvectionAnalytical>` or
  :py:class:`AdvectionInterpolation <lagrbfpy.AdvectionInterpolation>` 
  classes which provides a higher level interface.

  .. note::
    Calling this class will automatically create a new C context and initialize it with the provided options.
    To perform interpolation and/or advection in the same context i.e., same domain with same options, do not
    create a new instance of this class, but rather use the same instance to call the methods. 



  :param petsc4py.PETSc.Options petsc_options:
    A dictionary of PETSc options to configure the context.
    Can be created as

    .. code-block:: python

        petsc_options = PETSc.Options()

    If not provided, a new :py:class:`petsc4py.PETSc.Options` object will be created.

  :param int dim:
    The spatial dimension of the problem (default: ``3``).
  :param numpy.ndarray minCoords:
    The minimum coordinates of the partition of unity domain (default: array of zeros).
  :param numpy.ndarray maxCoords:
    The maximum coordinates of the partition of unity domain (default: array of ones).
  :param numpy.ndarray npartitions:
    The number of partitions of the partition of unity domain in each spatial direction (default: array of 4s).
  :param int nfields:
    The number of fields to interpolate (default: ``0``). 
    A vector field of size 3 would have ``nfields=3``, 
    a scalar field would have ``nfields=1``, etc.
    Many fields can be interpolated at once,
    ``nfields`` represents the total size of all fields to be interpolated.
  :param float overlap:
    The overlap between the partitions in the partition of unity method (default: ``1``).
  :param float shape_factor:
    The shape factor for the RBFs support (default: ``1``).
  :param int kernel:
    The type of kernel to use (default: ``2``: Wendland's C2 function).
    Available kernels are:

    - ``1``: Wendland's C0 function
    - ``2``: Wendland's C2 function
    - ``3``: Bump function
    - ``6``: Gaussian function

  :param bool advection:
    Whether to perform advection (default: False).
  :param int advection_scheme:
    The advection scheme to use (default: ``2``: RK4).
    Available schemes are:

    - ``0``: Euler
    - ``1``: RK2
    - ``2``: RK4
    - ``3``: Dormand-Prince 5(4)
  
  :param float advection_cfl:
    The Courant-Friedrichs-Lewy number for the advection scheme (default: ``0.5``).
  :param float rmin:
    The minimum radius for the advection in spherical domains (default: ``0``).
  :param float rmax:
    The maximum radius for the advection in spherical domains (default: ``1.0e132``).

  :Methods:
  """
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    self.data_filled:bool   = False
    self.interp_filled:bool = False
    self.PetscReal:np.dtype = np.dtype(PETSc.RealType)
    self.PetscRealCtype     = np.ctypeslib.as_ctypes_type(self.PetscReal)

    self.options = petsc_options
    if self.options is None:
      self.options:dict = PETSc.Options()

    self.output_path = kwargs.get("output_path", self.options.getString("output_path", "output"))
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path, exist_ok=True)
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
    """
    Return the C context pointer. This is a ``void *`` pointer that can be passed to the C functions.
    Its content should not be accessed directly in Python.

    :return: The C context pointer.
    :rtype: ctypes.c_void_p
    :raises RuntimeError: If the context has not been created or has been destroyed.
    """
    if self.ctx is None:
      raise RuntimeError("LagRBF context has not been created or has been destroyed.")
    return self.ctx
  
  def get_stored_refs(self) -> dict:
    """
    Return a dictionary of stored references to Python objects. 
    This is useful to prevent garbage collection of objects that are used in the C context.

    :return: A dictionary of stored references.
    :rtype: dict
    """
    return self.refs
  
  def get_ref(self, key:str):
    """
    Get a stored reference to a Python object by key.

    :param str key: The key of the stored reference.
    :return: The stored reference.
    :raises KeyError: If the key does not exist in the stored references.
    """
    ref = self.refs.get(key, None)
    if ref is None:
      raise KeyError(f"No reference found for key '{key}'.")
    return ref

  def store_ref(self, **objs):
    """
    Store references to Python objects to prevent garbage collection.

    :param objs: 
      Keyword arguments representing the objects to store.
      Example: 
      
      .. code-block:: python

          store_ref(advection_fn=my_function, advection_data=my_data)
    
    These objects can later be retrieved using :py:meth:`get_ref` with the corresponding keys.
    """
    self.refs.update(objs)

  # lets it be passed directly as c_void_p
  def __index__(self) -> ctypes.c_void_p:
    return self.ctx
  
  def __bool__(self) -> bool:
    return self.ctx is not None
  
  def __del__(self) -> None:
    """
    Call the C function to clear the structure in memory when the Python object is deleted.
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
    return c_arr.reshape((array.size,))
  
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
    """
    Fill the C swarm with the provided coordinates, values, and point indices.

    :param numpy.ndarray coords:
      A 2D array of shape ``(npoints, dim)`` containing the coordinates of the points to fill in the swarm.
    :param numpy.ndarray values:
      A 2D array of shape ``(npoints, nfields)`` containing the values of the fields at the points to fill in the swarm.
      If None, no values will be filled.
    :param numpy.ndarray pidx:
      A 1D array of shape ``(npoints,)`` containing the unique point indices of the points to fill in the swarm.
      If None, no point indices will be filled.
    :param bool is_data:
      If ``True``, fills the swarm corresponding to data points; if ``False``, fills the swarm corresponding to interpolation points.
    :raises ValueError: If coords is None or if the shapes of values or pidx do not match the number of points in coords.
    :raises RuntimeError: If the C function call fails, see PETSc error for details.
    """
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
    """
    Fill the swarm corresponding to data points with the provided coordinates, values, and point indices.
    Calls :py:meth:`fill_swarm` with ``is_data=True``.

    :param numpy.ndarray coords:
      A 2D array of shape ``(npoints, dim)`` containing the coordinates of the data points to fill in the swarm.
    :param numpy.ndarray values:
      A 2D array of shape ``(npoints, nfields)`` containing the values of the fields at the data points to fill in the swarm.
      If None, no values will be filled.
    :param numpy.ndarray pidx:
      A 1D array of shape ``(npoints,)`` containing the unique point indices of the data points to fill in the swarm.
      If None, no point indices will be filled.
    """
    self.fill_swarm(coords=coords, values=values, pidx=pidx, is_data=True)
    self.data_filled = True
    return
  
  def fill_points(self, coords:np.ndarray, values:np.ndarray|None=None, pidx:np.ndarray|None=None) -> None:
    """
    Fill the swarm corresponding to interpolation points with the provided coordinates, values, and point indices.
    Calls :py:meth:`fill_swarm` with ``is_data=False``.
    
    :param numpy.ndarray coords:
      A 2D array of shape ``(npoints, dim)`` containing the coordinates of the interpolation points to fill in the swarm.
    :param numpy.ndarray values:
      A 2D array of shape ``(npoints, nfields)`` containing the values of the fields at the interpolation points to fill in the swarm.
      If None, no values will be filled.
    :param numpy.ndarray pidx:
      A 1D array of shape ``(npoints,)`` containing the unique point indices of the interpolation points to fill in the swarm.
      If None, no point indices will be filled.
    """
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
    """
    Write the coordinates, field values, and point indices of the swarm to the provided arrays.
    This method is used to retrived the data from the C side to the Python side.
    In parallel, all ranks have access to the full data.

    :param numpy.ndarray coords:
      A 2D array of shape ``(npoints, dim)`` to hold the coordinates of the points in the swarm.
    :param numpy.ndarray fvals:
      A 2D array of shape ``(npoints, nfields)`` to hold the values of the fields at the points in the swarm.
    :param numpy.ndarray pidx:
      A 1D array of shape ``(npoints,)`` to hold the unique point indices of the points in the swarm.
    :param str fieldname:
      The name of the field to be outputed in the C swarm. Default is ``"fields"`` 
      and should not be modified except if you know what you are doing.
    :param bool is_data:
      If ``True``, outputs the swarm corresponding to data points; if ``False``, outputs the swarm corresponding to interpolation points.
    :raises ValueError: 
      If ``coords``, ``fvals``, and ``pidx`` are all None, 
      or if their shapes do not match the number of points in the swarm.
    :raises RuntimeError: 
      If the C function call fails, see 
      `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details.
    
    :Example usage:
      .. code-block:: python

        # Create empty arrays to hold the output data
        coords_out = np.empty((npoints, dim), dtype=np.float64)
        fvals_out  = np.empty((npoints, nfields), dtype=np.float64)
        pidx_out   = np.empty((npoints,), dtype=np.int64)

        # Call the output method to fill the arrays with data from the C swarm
        ctx.output(coords=coords_out, fvals=fvals_out, pidx=pidx_out, fieldname="fields")

      Unwanted fields can be set to ``None`` or simply ignored.

    """
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
    """
    Save points and their associated field values to a VTK file (``.vtp``) using the provided communicator.
    In parallel, only rank 0 will write the file to disk.

    :param petsc4py.PETSc.Comm comm:
      The PETSc communicator.
    :param str fname:
      The name of the output file (should end with ``.vtp``).
    :param dict fields_size:
      A dictionary mapping field names to their sizes (number of components). 
      Example: ``{"velocity": 3, "pressure": 1}``.
    :param lagrbfpy.PointContainer output_points:
      A :py:class:`PointContainer` class instance containing the coordinates, values, 
      and point indices of the points to be saved.
    :param str fieldname:
      The name of the fields to be retrieved from the C swarm, 
      do not change except if you know what you are doing. 
    :param bool is_data:
      If ``True``, saves the swarm corresponding to data points; if ``False``, saves the swarm corresponding to interpolation points.
    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_  
      for details.
    :raises ValueError: If the output file name does not end with ``.vtp`` or if the fields_size dictionary is empty.
    """
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
    """
    Call the C function to perform interpolation from the data swarm to the interpolation swarm.

    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details.
    """
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
    """
    Call the C function to reset the interpolation coefficients. 
    This is useful when performing multiple interpolations with different data or points.
    
    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details
    """
    ierr = cfuncs["lagrbf_reset_coefficients"](self.ctx)
    if ierr != 0:
      raise RuntimeError(f"lagrbf_reset_coefficients failed with error code {ierr}, see PETSc error for details.")
    return
  
  def __deprecated_set_advection_function(self, *args, **kwargs) -> None:
    """
    Set the advection function to be used for advecting points. 
    Two possibilities:

    1. 
      Provide the advection function and optional data directly as arguments:
      
      .. code-block:: python

          ctx.set_advection_function(my_advection_function, my_data)
    
    2.
      Get the advection function and optional data from stored references using keys.
      By default, it will look for ``"advection_fn"`` and ``"advection_data"`` keys in the stored references, 
      if not found it will look for keys provided in kwargs as ``"stored_fn_key"`` and ``"stored_data_key"``.
    
      
    :param args:
      The first argument should be the advection function (callable), 
      and the second argument (optional) should be the data to be passed to the advection function.
    :param kwargs:
      - ``stored_fn_key``: The key to look for in the stored references for the advection function (default: ``"advection_fn"``).
      - ``stored_data_key``: The key to look for in the stored references for the advection data (default: ``"advection_data"``).
    :raises ValueError: If no advection function is provided and no stored reference is found, or if the provided advection function is not callable.
    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details.
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
  
  def set_advection_function(self, func:ctypes.CFUNCTYPE, func_data:ctypes.Structure|None=None) -> None:
    if not callable(func):
      raise ValueError("Provided advection function is not callable.")
    if func_data is None:
      func_data = FunctionData(time=0.0, step=0)
    else:
      if not isinstance(func_data, ctypes.Structure):
        raise ValueError("Provided advection function data is not a ctypes.Structure.")
    self.store_ref(advection_fn=func)
    self.store_ref(advection_data=func_data)
    ierr = cfuncs["lagrbf_advection_function_batch"](
      self.ctx,
      func,
      ctypes.cast(ctypes.pointer(func_data), ctypes.c_void_p)
    )
    if ierr != 0:
      raise RuntimeError(f"lagrbf_advection_function_batch failed with error code {ierr}, see PETSc error for details.")    
    return

  def advect_analytical_c(self, time_start:float, time_end:float) -> None:
    """
    Call the C function to advect points from ``time_start`` to ``time_end`` 
    using the analytical advection function provided.

    :param float time_start: The starting time for the advection.
    :param float time_end: The ending time for the advection.
    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details.
    """
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
    """
    Call the C function to advect points from ``time_start`` to ``time_end`` 
    using interpolation from data points velocity to the advected points.

    :param float time_start: The starting time for the advection.
    :param float time_end: The ending time for the advection.
    :raises RuntimeError: 
      If the C function call fails, 
      see `PETSc error <https://petsc.org/release/include/petscsystypes.h.html#PetscErrorCode>`_ 
      for details.
    """
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