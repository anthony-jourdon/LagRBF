import os
import numpy as np
import pyvista as pvs

from .advection import Advection
from .utils.timeseries import timeseries_process
from .utils.points_container import PointContainer

from petsc4py import PETSc

def restart_from_pvd(
    comm:PETSc.Comm, 
    ctx:Advection, 
    pvdfile:str, 
    load_dir:str|None=None, 
    load_step:int|None=None
  ) -> PointContainer:
  # No specific load directory provided, use the output directory of the context
  if load_dir is None:
    load_dir = ctx.output_path

  timeseries = timeseries_process(os.path.join(load_dir, pvdfile))
  # if no step is provided, restart from the last step
  if load_step is None:
    load_step = -1

  if load_step == -1:
    ctx.start_idx = len(timeseries["step"]) - 1
  else:
    ctx.start_idx = load_step

  # load points
  rank = comm.getRank()
  if rank == 0:
    file  = timeseries["fname"][load_step]
    swarm:pvs.PolyData = pvs.read(os.path.join(load_dir, file))

    if "pidx" not in swarm.point_data:
      pidx   = np.arange(swarm.number_of_points, dtype=np.int64)
      coords = np.float64(swarm.points)
    else:
      # this is done to insure that even if some points are missing in the swarm, 
      # we still have a unique index for each point
      pid = swarm.point_data["pidx"]
      pidx = np.ones((np.max(pid),), dtype=np.int64)
      pidx[0:swarm.number_of_points] = pid
      coords = np.zeros((np.max(pid), 3), dtype=np.float64)
      coords[0:swarm.number_of_points, :] = np.float64(swarm.points)

    points = PointContainer(
      coords=np.float64(coords),
      pidx=np.int64(pidx)
    )
  else:
    points = PointContainer(
      coords=np.zeros((0, 3), dtype=np.float64),
      pidx=np.zeros((0,), dtype=np.int64)
    )
  return points