import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from petsc4py import PETSc

def wrap_ndpointer(dtype=None, ndim=None, shape=None, flags=None):
  base = ndpointer(dtype=dtype, ndim=ndim, shape=shape, flags=flags)
  class WrappedPointer(base):
    @classmethod
    def from_param(cls, obj):
      if obj is None:
        return obj
      return super().from_param(obj)
  return WrappedPointer

def load_shared_lib():
  libdir   = os.path.join(os.path.dirname(__file__),"..","..",os.environ["PETSC_ARCH"],"lib")
  clibname = "liblagrangian.so"
  clib = ctypes.cdll.LoadLibrary(os.path.join(libdir, clibname))
  return clib

def register_functions(clib:ctypes.CDLL):
  petsc_int_c      = np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType))
  petsc_int64_c    = np.dtype(np.int64)
  petsc_real_dtype = np.dtype(PETSc.RealType)
  ###################################################################
  # Stand alone interpolation function
  ###################################################################
  cfuncs = {}
  cfunc = clib.lagrbf_interpolate_driver
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
  cfuncs["lagrbf_interpolate_driver"] = cfunc

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

  ###################################################################
  # Swarm filling function
  ###################################################################
  cfunc = clib.lagrbf_fill_swarm
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,                                    # void *ctx
    ctypes.c_bool,                                      # PetscBool is_data
    petsc_int_c,                                        # PetscInt npoints
    ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"),  # PetscReal *coords
    wrap_ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"),  # PetscReal *values
    wrap_ndpointer(petsc_int64_c, flags="C_CONTIGUOUS") # PetscInt64 *pidx
  ]
  cfuncs["lagrbf_fill_swarm"] = cfunc

  ###################################################################
  # LagRBF interpolation function
  ###################################################################
  cfunc = clib.lagrbf_interpolation
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [ctypes.c_void_p] # void *ctx
  cfuncs["lagrbf_interpolation"] = cfunc

  ###################################################################
  # LagRBF output function
  ###################################################################
  cfunc = clib.lagrbf_output
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,                                        # void *ctx
    ctypes.c_bool,                                          # PetscBool is_data
    petsc_int_c,                                            # PetscInt npoints
    wrap_ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *coords
    wrap_ndpointer(petsc_real_dtype, flags="C_CONTIGUOUS"), # PetscReal *fields
    wrap_ndpointer(petsc_int64_c, flags="C_CONTIGUOUS"),    # PetscInt64 *pidx
    ctypes.c_char_p                                         # char *options_file
  ]
  cfuncs["lagrbf_output"] = cfunc

  ###################################################################
  # LagRBF analytical velocity advection function
  ###################################################################
  cfunc = clib.lagrbf_avect_analytical
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,  # void *ctx
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType)), # PetscReal time
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))  # PetscReal time_next 
  ]
  cfuncs["lagrbf_avect_analytical"] = cfunc

  ###################################################################
  # LagRBF advection function registering
  ###################################################################
  cfunc = clib.lagrbf_advection_function
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,
    ctypes.CFUNCTYPE(
      ctypes.c_int, 
      ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))), 
      ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))),
      ctypes.c_void_p
    ),
    ctypes.c_void_p
  ]
  cfuncs["lagrbf_advection_function"] = cfunc

  ###################################################################
  # LagRBF advection batch function registering
  ###################################################################
  cfunc = clib.lagrbf_advection_function_batch
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,
    ctypes.CFUNCTYPE(
      ctypes.c_int, 
      petsc_int_c,
      petsc_int_c,
      ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))), 
      ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))),
      ctypes.c_void_p
    ),
    ctypes.c_void_p
  ]
  cfuncs["lagrbf_advection_function_batch"] = cfunc

  ###################################################################
  # LagRBF interpolated velocity advection function
  ###################################################################
  cfunc = clib.lagrbf_advect_interpolate
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p,  # void *ctx
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType)), # PetscReal time
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.RealType))  # PetscReal time_next 
  ]
  cfuncs["lagrbf_avect_interpolate"] = cfunc

  ###################################################################
  # LagRBF reset interpolation coefficients
  ###################################################################
  cfunc = clib.lagrbf_reset_coefficients
  cfunc.restype = ctypes.c_int # PetscErrorCode
  cfunc.argtypes = [
    ctypes.c_void_p  # void *ctx
  ]
  cfuncs["lagrbf_reset_coefficients"] = cfunc

  return cfuncs


def load_cfuncs():
  clib = load_shared_lib()
  cfuncs = register_functions(clib)
  return cfuncs