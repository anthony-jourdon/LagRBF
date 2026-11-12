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
  file:    __init__.py

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


import sys, petsc4py
import warnings
petsc4py.init(sys.argv) # maybe this should be called in scripts, I don't know yet

def print(*args):
  petsc4py.PETSc.Sys.Print(*args)

from . import utils
from . import petsc_io
from . import swarm_utils
from . import interface
from . import interpolation
from . import advection
from . import restart

from .interface import cfuncs
from .utils.points_container import PointContainer, fill_datafields
from .utils.halton_seq import halton_sequence
from .swarm_utils import PointSwarm
from .utils.timeseries import timeseries_process, timeseries_compare, timeseries_write_step, timeseries_write_new, timeseries_append, timeseries_write
from .petsc_io.petscvec_loader import PetscVecLoadFromFile, PetscVecLoadBlockSizeFromFile, PetscVecLoad, PetscISLoadFromFile, PetscMatDenseLoadFromFile
from .petsc_io.petscvec_writer import PetscVecWriteToFile, PetscVecWriteBlockSizeToFile, PetscVecWrite, PetscISWriteToFile
from .interface.context import LagRBFCtx
from .interface.function_pointer import FunctionPointer, FunctionData
from .interpolation import LagRBFInterpolator
from .advection import Advection, AdvectionAnalytical, AdvectionInterpolation
from .restart import restart_from_pvd

_ANALYTICAL_FLOW_AVAILABLE = True
try:
  from .analytical_flow import HarmonicFlow, ToroidalFlow, PoloidalFlow, FlowData, toroidal_poloidal_flow, Ctoroidal_poloidal_flow, Ctoroidal_poloidal_flow_batch
except ModuleNotFoundError as exc:
  if exc.name == "sphericart":
    _ANALYTICAL_FLOW_AVAILABLE = False
    warnings.warn(
      "Optional dependency 'sphericart' is not installed. "
      "Analytical flow helpers and advection_benchmark.py are unavailable, "
      "but the rest of lagrbfpy will continue to work.",
      UserWarning,
      stacklevel=2,
    )
  else:
    raise

__all__ = [
  "cfuncs",
  "fill_datafields",
  "PointContainer",
  "halton_sequence_scalar",
  "halton_sequence_vector",
  "halton_sequence",
  "timeseries_process",
  "timeseries_compare",
  "timeseries_write_step",
  "timeseries_write_new",
  "timeseries_append",
  "timeseries_write",
  "PetscVecLoadFromFile",
  "PetscVecLoadBlockSizeFromFile",
  "PetscVecLoad",
  "PetscISLoadFromFile",
  "PetscMatDenseLoadFromFile",
  "PetscVecWriteToFile",
  "PetscVecWriteBlockSizeToFile",
  "PetscVecWrite",
  "PetscISWriteToFile",
  "LagRBFCtx",
  "FunctionPointer",
  "FunctionData",
  "PointSwarm",
  "LagRBFInterpolator",
  "Advection",
  "AdvectionAnalytical",
  "AdvectionInterpolation",
]

if _ANALYTICAL_FLOW_AVAILABLE:
  __all__ += [
    "HarmonicFlow",
    "ToroidalFlow",
    "PoloidalFlow",
    "FlowData",
    "toroidal_poloidal_flow",
    "Ctoroidal_poloidal_flow",
    "Ctoroidal_poloidal_flow_batch",
  ]