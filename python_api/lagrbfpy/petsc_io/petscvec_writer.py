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
  file:    petscvec_writer.py

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

def PetscISWriteToFile(fname:str, iset:np.ndarray):
  io  = PetscBinaryIO.PetscBinaryIO()
  iset_view = iset.view(PetscBinaryIO.IS)
  io.writeBinaryFile(fname, [iset_view,])
  return