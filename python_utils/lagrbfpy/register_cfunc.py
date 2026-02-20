import os
from numpy.ctypeslib import ndpointer
import ctypes

def load_shared_lib():
  libdir   = os.path.join(os.path.dirname(__file__),"..","..",os.environ["PETSC_ARCH"],"lib")
  clibname = "liblagrangian.so"
  clib = ctypes.cdll.LoadLibrary(os.path.join(libdir, clibname))
  return clib

def register_functions(clib:ctypes.CDLL):
  cfuncs = {}
  # function lagrbf_interpolate() in src/interface.c
  cfunc = clib.lagrbf_interpolate
  cfunc.restype = None
  cfunc.argtypes = [
    ctypes.c_int,                                     # int ndata
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # double *data_coords
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # double *data_values
    ctypes.c_int,                                     # int npoints
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # double *points_coords
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # double *points_values
    ctypes.c_char_p                                   # char *options_file
  ]
  cfuncs["lagrbf_interpolate"] = cfunc
  return cfuncs

def load_cfuncs():
  clib = load_shared_lib()
  cfuncs = register_functions(clib)
  return cfuncs