import os
import numpy as np
import lagrbfpy as lrb
from petsc4py import PETSc

from .interface import LagRBFCtx
from .utils.points_container import PointContainer

class LagRBFInterpolator(LagRBFCtx):
  def __init__(self, petsc_options:dict|None=None, **kwargs):
    super().__init__(petsc_options=petsc_options, **kwargs)
    return
  
  def interpolate(
      self, 
      data:PointContainer,
      points:PointContainer,
      output:PointContainer,
      fieldname:str="fields",
    ):
    self.fill_data(
      coords=data.coords,
      values=data.values,
      pidx=data.pidx
    )
    self.fill_points(
      coords=points.coords,
      values=points.values,
      pidx=points.pidx
    )
    self.interpolate_c()
    self.output(
      coords=output.coords,
      fvals=output.values,
      pidx=output.pidx,
      fieldname=fieldname,
      is_data=False
    )
    self.reset_coefficients()
    return