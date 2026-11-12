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
  file:    petscvec_loader.py

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

import os,sys
import numpy as np
# Add petsc binary IO to the system path
sys.path.append(os.path.join(os.environ.get('PETSC_DIR'),'lib','petsc','bin'))
import PetscBinaryIO

def PetscVecLoadFromFile(fname, **kwags):
  io = PetscBinaryIO.PetscBinaryIO(**kwags) # Instantiate a petsc binary loader
  with open(fname) as fp:
    objecttype = io.readObjectType(fp)
    v = io.readVec(fp)
  return v

def PetscVecLoadBlockSizeFromFile(fname):
  with open(fname) as fp:
    line = fp.readline()
    if not line.startswith('-vecload_block_size'):
      raise ValueError(f'Expected first line to be "-vecload_block_size <size>", got: {line}')
    block_size = int(line.split()[1])
  return block_size

def PetscVecLoad(fname, order='C', **kwargs):
  v = PetscVecLoadFromFile(fname, **kwargs)
  try:
    bs = PetscVecLoadBlockSizeFromFile(fname + '.info')
    v = np.reshape(v, (-1, bs), order=order)
  except FileNotFoundError:
    return v
  return v

def PetscISLoadFromFile(fname,**kwargs):
  io = PetscBinaryIO.PetscBinaryIO(**kwargs) # Instantiate a petsc binary loader
  with open(fname) as fp:
    objecttype = io.readObjectType(fp)
    v = io.readIS(fp)
  return v

def PetscMatDenseLoadFromFile(fname, **kwargs):
  io = PetscBinaryIO.PetscBinaryIO(**kwargs) # Instantiate a petsc binary loader
  with open(fname) as fp:
    objecttype = io.readObjectType(fp)
    m = io.readMatDense(fp)
  return m

def test():
  import matplotlib.pyplot as plt
  root = os.path.dirname(__file__)
  coor_fname = os.path.join(root,'../output/swarm_interp_coords.pbvec')
  coor_bsinfo = coor_fname+'.info'
  field_fname = os.path.join(root,'../output/swarm_interp_fields.pbvec')
  field_bsinfo = field_fname+'.info'

  coor_bs = PetscVecLoadBlockSizeFromFile(coor_bsinfo)
  print(f"Coordinate block size: {coor_bs}")
  coor = PetscVecLoadFromFile(coor_fname)
  coor = np.reshape(coor, (-1, coor_bs))
  print(f"Coordinate shape: {coor.shape}")
  
  field_bs = PetscVecLoadBlockSizeFromFile(field_bsinfo)
  print(f"Field block size: {field_bs}")
  field = PetscVecLoadFromFile(field_fname)
  field = np.reshape(field, (-1, field_bs))
  print(f"Field shape: {field.shape}")
  
  fig,ax = plt.subplots(1, field_bs)
  cb = [None]*field_bs
  for i in range(field_bs):
    cb[i] = ax[i].scatter(coor[:,0], coor[:,1], c=field[:,i], s=15, cmap='viridis')
    ax[i].set_title(f'Field {i}')
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    ax[i].set_aspect('equal')
    plt.colorbar(cb[i], ax=ax[i], label=f'Field {i} value', orientation='horizontal')

  plt.tight_layout()
  plt.show()

  return

if __name__ == '__main__':
  test()