import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from petsc4py import PETSc

def load_shared_lib():
  libdir   = os.path.join(os.path.dirname(__file__),"..","..",os.environ["PETSC_ARCH"],"lib")
  clibname = "liblagrangian.so"
  clib = ctypes.cdll.LoadLibrary(os.path.join(libdir, clibname))
  return clib

def register_functions(clib:ctypes.CDLL):
  petsc_int_c = np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType))
  petsc_real_dtype = np.dtype(PETSc.RealType)
  ###################################################################
  # Stand alone interpolation function
  ###################################################################
  cfuncs = {}
  # function lagrbf_interpolate() in src/interface.c
  cfunc = clib.lagrbf_interpolate
  cfunc.restype = None
  cfunc.argtypes = [
    petsc_int_c,                                       # PetscInt ndata_local
    ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *data_coords
    ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *data_values
    petsc_int_c,                                       # PetscInt npoints_local
    ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *points_coords
    ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *points_values
    petsc_int_c,                                       # PetscInt npoints_global
    ctypes.c_char_p                                    # char *options_file
  ]
  cfuncs["lagrbf_interpolate"] = cfunc

  ###################################################################
  # LagRBF context creation
  ###################################################################
  cfunc = clib.lagrbf_create
  cfunc.restype = ctypes.c_void_p # void *ctx
  cfunc.argtypes = [] # no arguments
  cfuncs["lagrbf_create"] = cfunc

  ###################################################################
  # LagRBF context cleanup
  ###################################################################
  cfunc = clib.lagrbf_destroy
  cfunc.restype = None
  cfunc.argtypes = [ctypes.c_void_p] # void *ctx
  cfuncs["lagrbf_destroy"] = cfunc

  return cfuncs

def load_cfuncs():
  clib = load_shared_lib()
  cfuncs = register_functions(clib)
  return cfuncs