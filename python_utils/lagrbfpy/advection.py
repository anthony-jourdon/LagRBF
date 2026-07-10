import os
import ctypes
import numpy as np
from petsc4py import PETSc
from pyvista import PolyData

from .interface import LagRBFCtx
from .utils.points_container import PointContainer
from .utils.timeseries import timeseries_write

class Advection(LagRBFCtx):
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
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
    return
  
  def set_advection_function(self, func:ctypes.CFUNCTYPE, func_data:ctypes.Structure|None=None):
    super().set_advection_function(func, func_data)
    return
  
  def advect(self, points:PointContainer, output:PointContainer, comm:PETSc.Comm, prefix:str|None=None):
    rank = comm.getRank()
    self.fill_points( coords=points.coords, values=points.values, pidx=points.pidx )
    for step in range(self.nsteps-1):
      PETSc.Sys.Print(f"\n[[ STEP: {step} ]]: Advecting points from time {self.time_steps[step]:.6e} to {self.time_steps[step+1]:.6e}")
      # advect points from time_steps[step] to time_steps[step+1]
      self.advect_analytical_c(
        self.time_steps[step  ],
        self.time_steps[step+1]
      )
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
      # append the time series file
      if rank == 0:
        timeseries = {
          "time": [f"{self.time_steps[step+1]:.6e}"], # store time as string
          "step": [f"{(self.steps[step+1]):05d}"], # store step number as string
        }
        timeseries_write(fname=os.path.join(self.output_path, self.pvdfname), timeseries=timeseries, prefix=prefix, extension="vtp", erase=False)
    return

class AdvectionInterpolation(Advection):
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
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
    if not callable(get_data):
      raise ValueError("get_data must be a callable function that returns a PointContainer.")
    rank = comm.getRank()
    self.fill_points( coords=points.coords, values=points.values, pidx=points.pidx )
    for step in range(self.nsteps):
      PETSc.Sys.Print(f"\n[[ STEP: {step} ]]: Advecting points from time {self.time_steps[step]:.6e} to {self.time_steps[step+1]:.6e}")
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
    
      if step == self.nsteps-1:
        break
      # advect points from time_steps[step] to time_steps[step+1]
      self.advect_interpolate_c(
        self.time_steps[step  ], 
        self.time_steps[step+1]
      )
    return