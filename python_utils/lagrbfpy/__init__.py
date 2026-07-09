
import sys, petsc4py
petsc4py.init(sys.argv) # maybe this should be called in scripts, I don't know yet

from . import register_cfunc
cfuncs = register_cfunc.load_cfuncs()

from . import utils
from . import petsc_io
from . import swarm_utils
from . import interface
from . import interpolation

def print(*args):
  petsc4py.PETSc.Sys.Print(*args)

from .utils.points_container import PointContainer
from .utils.halton_seq import halton_sequence_scalar, halton_sequence_vector, halton_sequence
from .swarm_utils import PointSwarm
from .utils.timeseries import timeseries_process, timeseries_compare, timeseries_write_step, timeseries_write_new, timeseries_append, timeseries_write
from .petsc_io.petscvec_loader import PetscVecLoadFromFile, PetscVecLoadBlockSizeFromFile, PetscVecLoad, PetscISLoadFromFile, PetscMatDenseLoadFromFile
from .petsc_io.petscvec_writer import PetscVecWriteToFile, PetscVecWriteBlockSizeToFile, PetscVecWrite, PetscISWriteToFile
from .interface.context import LagRBFCtx
from .analytical_flow import HarmonicFlow, ToroidalFlow, PoloidalFlow, FlowData, toroidal_poloidal_flow, Ctoroidal_poloidal_flow, Ctoroidal_poloidal_flow_batch
from .interpolation import LagRBFInterpolator

__all__ = [
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
  "PointSwarm",
  "HarmonicFlow",
  "ToroidalFlow",
  "PoloidalFlow",
  "FlowData",
  "toroidal_poloidal_flow",
  "Ctoroidal_poloidal_flow",
  "Ctoroidal_poloidal_flow_batch",
  "LagRBFInterpolator"
]