
import sys, petsc4py
petsc4py.init(sys.argv) # maybe this should be called in scripts, I don't know yet

from .utils    import *
from .petsc_io import *

from .register_cfunc  import *
from .interpolation   import *

cfuncs = load_cfuncs()
