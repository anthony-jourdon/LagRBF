import os
import ctypes
import numpy as np
from petsc4py import PETSc
from pyvista import PolyData

from .interface import LagRBFCtx
from .utils.points_container import PointContainer
from .utils.timeseries import timeseries_write

class Advection(LagRBFCtx):
  """
  General class to perform advection of Lagrangian points using :py:class:`LagRBFCtx <lagrbfpy.LagRBFCtx>` methods.
  This class is not meant to be used directly, but rather to be subclassed for specific advection methods.

  Advection related parameters from :py:class:`LagRBFCtx <lagrbfpy.LagRBFCtx>` are:

  :param int advection_scheme:
    The advection scheme to use.

    - 0: Forward Euler
    - 1: Runge-Kutta 2nd order
    - 2: Runge-Kutta 4th order
    - 3: Dormand-Prince 5(4)
  
  :param float advection_cfl:
    The CFL number to use for the advection scheme. 
    This is used to compute the time step size for the advection scheme.
  :param float time_start:
    The starting time for the advection.
  :param float time_end:
    The ending time for the advection.
  :param int nsteps:
    The number of time steps to use for the advection.
    
    .. note:: 
      
      This is not used to determine the time step size, but rather to determine the number 
      of steps to input data and output results.

  :param numpy.ndarray time_steps:
    An array of time steps to use for the advection e.g., ``[0.0, 1.5, 10.0, 50.0]``.
    If provided, this will override the time_start, time_end and nsteps parameters.

    .. note::
      The step between two consecutive time steps does not determine the time step size 
      for the advection scheme, but rather the time at which to input data and output results.

  :param numpy.ndarray steps:
    An array of step numbers to use for the advection e.g., ``[0, 1, 2, 3]``.
    If provided, this will override the default step numbers which are ``[0, 1, 2, ..., nsteps-1]``.
  :param str pvdfilename:
    The name of the output PVD file to write the time series of the advected points. 
    Default is ``"timeseries-advection.pvd"``.

  :param dict petsc_options:
      A dictionary of PETSc options to configure the context.
      Can be created as

      .. code-block:: python

          petsc_options = PETSc.Options()

      If not provided, a new `PETSc.Options` object will be created.
  
  """
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    kwargs["advection"]        = True
    kwargs["advection_scheme"] = kwargs.get("advection_scheme", 2)
    kwargs["advection_cfl"]    = kwargs.get("advection_cfl", 0.3)
    kwargs["dt"]               = kwargs.get("dt", 1e-7)
    super().__init__(petsc_options=petsc_options, **kwargs)
  
    time_start = kwargs.pop("time_start", None)
    time_end   = kwargs.pop("time_end", None)
    nsteps     = kwargs.pop("nsteps", None)
    time_steps = kwargs.pop("time_steps", None)
    steps      = kwargs.pop("steps", None)
    # if time_steps is provided we will use it, otherwise we generate it from time_start, time_end and nsteps
    if time_steps is None:
      if time_start is None or time_end is None or nsteps is None:
        serr = "time_start, time_end and nsteps must be provided if time_steps is not provided.\n"
        serr += f"Found: time_start = {time_start}, time_end = {time_end}, nsteps = {nsteps}, time_steps = {time_steps}"
        raise ValueError(serr)
      self.time_steps = np.linspace(time_start, time_end, nsteps, dtype=np.float64)
      self.nsteps     = nsteps
    else:
      self.time_steps = np.asarray(time_steps, dtype=np.float64)
      self.nsteps     = self.time_steps.shape[0]
    
    if steps is None:
      self.steps = np.arange(self.nsteps, dtype=np.int32)
    else:
      self.steps = np.asarray(steps, dtype=np.int32)
      if self.steps.shape[0] != self.nsteps:
        serr = f"steps must have the same length as time_steps.\n"
        serr += f"Found: steps.shape[0] = {self.steps.shape[0]}, time_steps.shape[0] = {self.nsteps}"
        raise ValueError(serr)

    self.pvdfname = kwargs.get("pvdfilename", "timeseries-advection.pvd")
    return
  
  def advect(self):
    raise NotImplementedError("The advect method must be implemented in a subclass.")
  
class AdvectionAnalytical(Advection):
  """
  Class to perform advection of Lagrangian points using an analytical velocity field.
  See :py:class:`Advection <lagrbfpy.Advection>` for more details on the parameters.
  """
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
    return
  
  def set_advection_function(self, func:ctypes.CFUNCTYPE, func_data:ctypes.Structure|None=None):
    """
    Register the advection function to be used for advecting points.

    :param ctypes.CFUNCTYPE func:
      A Python function that computes the velocity field at given coordinates.
      The function must have the following signature:

      .. code-block:: python

        @ctypes.CFUNCTYPE(
          ctypes.c_int,                                         # return error code (0 for success, non-zero for failure)
          np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # npoints
          np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # dim
          ctypes.POINTER(                                       # coords
            np.ctypeslib.as_ctypes_type(
              np.dtype(PETSc.RealType)
            )
          ),
          ctypes.POINTER(                                       # velocity
            np.ctypeslib.as_ctypes_type(
              np.dtype(PETSc.RealType)
            )
          ),
          ctypes.c_void_p                                       # data_ptr
        )
        def my_advection_function(npoints, dim, coords, velocity, data_ptr):
          data = ctypes.cast(
            data_ptr,
            ctypes.POINTER(MyDataStructure)
          ).contents

          X = np.ctypeslib.as_array(coords,   shape=(npoints, dim))
          V = np.ctypeslib.as_array(velocity, shape=(npoints, dim))

          # code to compute the velocity field at the given coordinates X using the data in data_ptr
          V[:, :] = my_python_function(X, data)
          return 0

    :param ctypes.Structure func_data:
      A ctypes Structure containing any additional data needed by the advection function.
      This data will be passed to the advection function as a pointer.

      .. code-block:: python

        class MyDataStructure(ctypes.Structure):
          _fields_ = [
            ("param1", ctypes.c_double),
            ("param2", ctypes.c_int),
            # add more fields as needed
          ]
      
      Any ctypes compatible data structure can be used.
      The user is left responsible for ensuring that the data structure is 
      properly initialized and compatible with the advection function.
    """
    super().set_advection_function(func, func_data)
    return
  
  def advect_step(self, step:int, output:PointContainer, comm:PETSc.Comm, prefix:str=""):
    """
    Advect points for a single time step using the registered advection function and the provided timesteps at class instantiation.
    The advected points are saved to disk.

    :param int step:
      The index of the time step to perform advection for.
    :param lagrbfpy.PointContainer output:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance to hold the output of the advected points. 
      Outputs points advected coordinates, their velocity, and their unique point indices.
    :param PETSc.Comm comm:
      The PETSc communicator to use.
    :param str prefix:
      A prefix to add to the output file names. 
      Files are named as 

      .. code-block:: python

        step{step_number:05d}{prefix}.vtp

      Default is an empty string.  
    """
    time      = self.time_steps[step]
    time_next = self.time_steps[step+1]
    PETSc.Sys.Print(f"\n[[ STEP: {step} ]]: Advecting points from time {time:.6e} to {time_next:.6e}")
    # update the time and step in the func_data structure if they exist
    func_data = self.get_ref("advection_data")
    if hasattr(func_data, "time"):
      func_data.time = np.float64(time)
    if hasattr(func_data, "step"):
      func_data.step = np.int32(self.steps[step])
    # advect points from time_steps[step] to time_steps[step+1]
    self.advect_analytical_c(time, time_next)
    # output
    fname = f"step{(self.steps[step+1]):05d}{prefix}.vtp"
    self.save(
      comm, 
      fname=os.path.join(self.output_path, fname),
      fields_size={"velocity": self.dim},
      output_points=output,
      fieldname="velocity",
      is_data=False,
    )
    # append the timeseries file
    rank = comm.getRank()
    if rank == 0:
      timeseries = {
        "time": [f"{self.time_steps[step+1]:.6e}"], # store time as string
        "step": [f"{(self.steps[step+1]):05d}"], # store step number as string
      }
      timeseries_write(fname=os.path.join(self.output_path, self.pvdfname), timeseries=timeseries, prefix=prefix, extension="vtp", erase=False)
    return

  def advect(self, points:PointContainer, output:PointContainer, comm:PETSc.Comm, prefix:str=""):
    """
    Advect points using the registered advection function and the provided timesteps at class instantiation.
    The advected points are saved to disk at each time step.

    .. note::
      Step 0 is the initial condition and is not saved to disk.
      The first output is saved at step 1, **after** the first advection step from time_steps[0] to time_steps[1].

    :param lagrbfpy.PointContainer points:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance containing the coordinates of the points to be advected. 
    :param lagrbfpy.PointContainer output:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance to hold the output of the advected points. 
      Outputs points advected coordinates, their velocity, and their unique point indices.
    :param PETSc.Comm comm:
      The PETSc communicator to use.
    :param str prefix:
      A prefix to add to the output file names. 
      Files are named as 

      .. code-block:: python

        step{step_number:05d}{prefix}.vtp

      Default is an empty string.  
    """
    try:
      _ = self.get_ref("advection_fn")
    except KeyError as e:
      raise RuntimeError("No advection function has been registered. Please use set_advection_function() to register an advection function before calling advect().") from e
    self.fill_points( coords=points.coords, values=points.values, pidx=points.pidx )
    for step in range(self.nsteps-1):
      self.advect_step(step, output, comm, prefix)
    return

class AdvectionInterpolation(Advection):
  """
  Class to perform advection of Lagrangian points using an interpolated velocity field.
  See :py:class:`Advection <lagrbfpy.Advection>` for more details on the parameters.
  """
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
    return
  
  def advect_step(
      self, 
      step:int, 
      get_data:callable, 
      data_args:dict, 
      output:PointContainer, 
      comm:PETSc.Comm, 
      fields:dict[str, int], 
      prefix:str=""
    ):
    """
    Advect points by interpolating the velocity field from a set of data points for a single time step.
    The advected points are saved to disk. 
    The output is done **before** advection and **after** interpolation

    :param int step:
      The index of the current step to perform advection for.
    :param callable get_data:
      A callable function that returns a :py:class:`PointContainer <lagrbfpy.PointContainer>` containing the 
      data points and their associated values to be used for interpolation at each time step. 
      The function must accept the step number as the first argument and the keyword arguments provided in 
      `data_args` and return a :py:class:`PointContainer <lagrbfpy.PointContainer>` instance.
    :param dict data_args:
      A dictionary of keyword arguments to be passed to the ``get_data()`` callable function. 
      This allows the user to provide any additional parameters needed to retrieve the data points 
      at each time step.

      .. note::
        The ``data_args`` dictionary can contain the keys ``"step"`` and ``"time"``, 
        which will be automatically updated to the current step and time for each call to ``get_data()``.

    :param lagrbfpy.PointContainer output:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance 
      to hold the output of the advected points. 
      Outputs points advected coordinates, their interpolated values, and their unique point indices.
    :param PETSc.Comm comm:
      The PETSc communicator to use.
    :param dict fields:
      A dictionary specifying the names and sizes of the fields to be interpolated at each time step. 
      The keys are the field names and the values are the sizes of the fields.
      Example: ``{"velocity": 3, "pressure": 1}`` for a 3D velocity field and a scalar pressure field.
    :param str prefix:
      A prefix to add to the output file names. 
      Files are named as

      .. code-block:: python

        step{step_number:05d}{prefix}.vtp

      Default is an empty string.

    """
    if step == self.nsteps-1:
      PETSc.Sys.Print(f"\n[[ STEP: {step} ]]: Last step, no advection performed, only outputting points and interpolated values at time {self.time_steps[step]:.6e}")
    else:
      PETSc.Sys.Print(f"\n[[ STEP: {step} ]]: Advecting points from time {self.time_steps[step]:.6e} to {self.time_steps[step+1]:.6e}")
    rank = comm.getRank()
    if "step" in data_args:
      data_args["step"] = np.int32(self.steps[step])
    if "time" in data_args:
      data_args["time"] = np.float64(self.time_steps[step])
    if "rank" in data_args:
      data_args["rank"] = np.int32(rank)
    data:PointContainer = get_data(**data_args)
    self.fill_data(coords=data.coords, values=data.values, pidx=data.pidx)
    # interpolate
    self.interpolate_c()
    # output
    fname = f"step{(self.steps[step]):05d}{prefix}.vtp"
    self.save(
      comm,
      fname=os.path.join(self.output_path, fname),
      fields_size=fields,
      output_points=output,
      fieldname="fields",
      is_data=False,
    ) 
    if rank == 0:
      timeseries = {
        "time": [f"{self.time_steps[step]:.6e}"], # store time as string
        "step": [f"{(self.steps[step]):05d}"], # store step number as string
      }
      timeseries_write(fname=os.path.join(self.output_path, self.pvdfname), timeseries=timeseries, prefix=prefix, extension="vtp", erase=False)
    # do not perform advection on the last step, only output the points and interpolated values
    if step == self.nsteps-1:
      return
    # advect points from time_steps[step] to time_steps[step+1]
    self.advect_interpolate_c(
      self.time_steps[step  ], 
      self.time_steps[step+1]
    )
    return

  def advect(
    self, 
    get_data:callable,
    data_args:dict, 
    points:PointContainer, 
    output:PointContainer, 
    comm:PETSc.Comm, 
    fields:dict[str, int],
    prefix:str=""
  ):
    """
    Advect points by interpolating the velocity field from a set of data points at each time step.
    The advected points are saved to disk at each time step. 

    .. note::
      The output is done **before** advection and **after** interpolation, therefore step 0 corresponds to
      the initial condition before any advection has been performed and the last step do not perform any 
      advection but only outputs the points and the values interpolated at their last advected location. 

    :param callable get_data:
      A callable function that returns a :py:class:`PointContainer <lagrbfpy.PointContainer>` containing the 
      data points and their associated values to be used for interpolation at each time step. 
      The function must accept the step number as the first argument and the keyword arguments provided in 
      `data_args` and return a :py:class:`PointContainer <lagrbfpy.PointContainer>` instance.
    :param dict data_args:
      A dictionary of keyword arguments to be passed to the ``get_data()`` callable function. 
      This allows the user to provide any additional parameters needed to retrieve the data points 
      at each time step.

      .. note::
        The ``data_args`` dictionary can contain the keys ``"step"`` and ``"time"``, 
        which will be automatically updated to the current step and time for each call to ``get_data()``.

    :param lagrbfpy.PointContainer points:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance 
      containing the coordinates of the points to be advected. 
    :param lagrbfpy.PointContainer output:
      A :py:class:`PointContainer <lagrbfpy.PointContainer>` class instance 
      to hold the output of the advected points. 
      Outputs points advected coordinates, their interpolated values, and their unique point indices.
    :param PETSc.Comm comm:
      The PETSc communicator to use.
    :param dict fields:
      A dictionary specifying the names and sizes of the fields to be interpolated at each time step. 
      The keys are the field names and the values are the sizes of the fields.
      Example: ``{"velocity": 3, "pressure": 1}`` for a 3D velocity field and a scalar pressure field.
    :param str prefix:
      A prefix to add to the output file names. 
      Files are named as

      .. code-block:: python

        step{step_number:05d}{prefix}.vtp

      Default is an empty string.

    """
    if not callable(get_data):
      raise ValueError("get_data must be a callable function that returns a PointContainer.")
    self.fill_points( coords=points.coords, values=points.values, pidx=points.pidx )
    for step in range(self.nsteps):
      self.advect_step(step, get_data, data_args, output, comm, fields, prefix=prefix)
    return