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
  file:    interpolation.py

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
import lagrbfpy as lrb
from petsc4py import PETSc

from .interface import LagRBFCtx
from .utils.points_container import PointContainer

class LagRBFInterpolator(LagRBFCtx):
  """
  Class to perform interpolation using Radial Basis Functions (RBF).
  This class provides methods to interpolate data from a set of points to 
  another set of points using RBFs. 
  It calls the C implementation :c:func:`RBFInterpolation` to perform the interpolation.

  .. warning::
    RBF interpolation is implemented without polynomial augmentation,
    therefore, no polynomial kernels should be used.

  :param petsc4py.PETSc.Options petsc_options: 
    A dictionary of PETSc options to configure the interpolation process.
    Can be created as 
    
    .. code-block:: python

        petsc_options = PETSc.Options()

  :param kwargs:
    Additional keyword arguments to configure the interpolation process.
    See :py:class:`LagRBFCtx <lagrbfpy.LagRBFCtx>` for more details on the available options.

  """
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
    """
    Interpolate data from a set of points to another set of points using Radial Basis Functions (RBF).
    Options are parsed from the `petsc_options` dictionary provided during initialization.

    .. note::
      Once the interpolation is complete, the coefficients are reset therefore
      several interpolations can be performed with the same context but different values and points.
      However, the interpolation parameters (e.g. kernel, polynomial degree, Partition of unity, etc.) 
      cannot be changed between calls to this method.

    :param lagrbfpy.PointContainer data:
      A :py:class:`PointContainer` class instance containing the coordinates and 
      values of the data points to be interpolated. 
    :param lagrbfpy.PointContainer points:
      A :py:class:`PointContainer` class instance containing the coordinates of the points
      where the data will be interpolated. If successively interpolating to the same points, 
      the :py:attr:`PointContainer.pidx` unique point indices should be provided.
    :param lagrbfpy.PointContainer output:
      A :py:class:`PointContainer` class instance to hold the output interpolated values
      at the specified points. The :py:attr:`PointContainer.pidx` unique point index is 
      outputed here and can be used for successive interpolations to the same points.
    :param str fieldname:
      The name of the field to be interpolated in the C swarm. 
      Default is ``"fields"`` and should not be modified except if you know what you are doing.
    
    """
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