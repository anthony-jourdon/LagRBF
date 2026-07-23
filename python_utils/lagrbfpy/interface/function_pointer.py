import ctypes
import numpy as np
from petsc4py import PETSc

class FunctionData(ctypes.Structure):
  """
  Class to hold user-defined data to be passed to a C callback function.
  Can also be used as a dataclass to hold parameters. 
  This class can be subclassed to add additional fields as needed. 
  The base class includes two fields: ``time`` and ``step``, 
  which can be used to track the simulation time and step number, respectively.
  """
  _fields_ = [
    ("time", ctypes.c_double),
    ("step", ctypes.c_int32),
  ]

class FunctionPointer:
  """
  Class to hold a Python function and its associated data, and create a C function pointer that can be passed to C code.
  The Python function defining the C callback must have the following prototype:

  .. code-block:: python

    def myfunction(npoints, dim, coords, velocity, data_ptr) -> int:
      # cast the data pointer to the appropriate type
      # MyFunctionData is a subclass of FunctionData or 
      # ctypes.Structure that holds user-defined parameters
      data = ctypes.cast(
        data_ptr,
        ctypes.POINTER(MyFunctionData)
      ).contents
      # do something with the data and the coordinates and velocities
      return 0

  where:

  - ``int npoints``: number of points

  - ``int dim``: dimension of the points

  - ``double* coords``: pointer to an array of coordinates

  - ``double* velocity``: pointer to an array of velocities

  - ``void* data_ptr``: pointer to user-defined data

  The user-defined data can be any subclass of :py:class:`ctypes.Structure`, 
  including the provided :py:class:`FunctionData` class. 
  The data can be used to pass parameters to the function.

  :param func: Python function to be called from C code. Must have the signature described above.
  :param ctypes.Structure data: 
    User-defined data to be passed to the function. 
    Can be any subclass of :py:class:`ctypes.Structure`.
    Default is None, in which case no data is passed to the function.

  The C function pointer can be accessed via the ``cfunc`` attribute,
  and the user-defined data can be accessed via the ``data`` attribute.
  """
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