import numpy as np
import PetscBinaryIO

def PetscVecWriteToFile(fname:str, v:np.ndarray):
  io  = PetscBinaryIO.PetscBinaryIO()
  vec = v.view(PetscBinaryIO.Vec)
  io.writeBinaryFile(fname, [vec,])
  return

def PetscVecWriteBlockSizeToFile(fname:str, v:np.ndarray):
  if v.ndim > 1:
    bs = v.shape[1]
  else:
    bs = 1
  with open(fname, 'w') as fp:
    fp.write(f'-vecload_block_size {bs}\n')
  return bs

def PetscVecWrite(fname:str, v:np.ndarray):
  bs = PetscVecWriteBlockSizeToFile(fname+".info", v)
  if bs > 1:
    v = np.reshape(v, (v.shape[0] * bs))
  PetscVecWriteToFile(fname, v)
  return