import os
import numpy as np
import lagrbfpy as lrb
from petsc4py import PETSc

from .interface import LagRBFCtx
from .swarm_utils import PointSwarm

class LagRBFInterpolator(LagRBFCtx):
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
    return
  
  def interpolate(
      self, 
      swarm_data:PointSwarm, 
      swarm_interp:PointSwarm, 
      out_coords:np.ndarray|None=None,
      out_fvals:np.ndarray|None=None,
      out_pidx:np.ndarray|None=None,
      fieldname:str="fields",
    ):
    self.fill_interpolate(swarm_data, swarm_interp)
    self.output(
      coords=out_coords,
      fvals=out_fvals,
      pidx=out_pidx,
      fieldname=fieldname,
      is_data=False
    )
    return