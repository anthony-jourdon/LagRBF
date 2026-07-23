import ctypes
import numpy as np
from petsc4py import PETSc

class FunctionData(ctypes.Structure):
  _fields_ = [
    ("time", ctypes.c_double),
    ("step", ctypes.c_int32),
  ]

class FunctionPointer:
  def __init__(self, func:callable, data:ctypes.Structure|None=None):
    if data is not None:
      if not isinstance(data, ctypes.Structure):
        serr =  "data must be a subclass of ctypes.Structure or FunctionData.\n"
        serr += "Build your own class from the helper class FunctionData, e.g.:\n"
        serr += "class MyFunctionData(FunctionData):\n"
        serr += "    _fields_ = [\n"
        serr +="         # put your parameters here (can be any ctypes type)\n"
        serr += "        ('param1', ctypes.c_double),\n"
        serr += "        ('param2', ctypes.c_int32),\n"
        serr += "    ]\n"
        raise TypeError(serr)
    if not callable(func):
      serr = "func must be a callable function with the signature myfunction(npoints, dim, coords, velocity, data_ptr)\n"
      raise TypeError(serr)
    self.func  = func
    self.data  = data
    self.cfunc = ctypes.CFUNCTYPE(
      ctypes.c_int,                                         # return type
      np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # npoints
      np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # dim
      ctypes.POINTER(
        np.ctypeslib.as_ctypes_type(
          np.dtype(PETSc.RealType)                          # coords
        )
      ),
      ctypes.POINTER(
        np.ctypeslib.as_ctypes_type(
          np.dtype(PETSc.RealType)                          # velocity
        )
      ),
      ctypes.c_void_p                                       # data pointer
    )(func)

def test():
  import inspect

  class MyFunctionData(FunctionData):
    _fields_ = [
      ("p1", ctypes.c_double),
      ("p2", ctypes.c_int32),
    ]
  
  def myfunction(npoints, dim, coords, velocity, data_ptr):
    data = ctypes.cast(
      data_ptr,
      ctypes.POINTER(MyFunctionData)
    ).contents
    X = np.ctypeslib.as_array(coords, shape=(npoints, dim))
    V = np.ctypeslib.as_array(velocity, shape=(npoints, dim))
    V[:, :] = X * data.p1 + data.p2
    return 0
  
  data = MyFunctionData(
    time=1.2, # inherited from FunctionData
    step=0,   # inherited from FunctionData
    p1=2.0, 
    p2=3
  )
  
  fp = FunctionPointer(myfunction, data)
  print(fp.cfunc)
  print(inspect.signature(fp.cfunc))

  for i in range(0,6):
    # update the data
    data.time += 0.1
    data.step += 1
    print(f"Step {data.step}, time {data.time}")
    

  return

if __name__ == "__main__":
  test()