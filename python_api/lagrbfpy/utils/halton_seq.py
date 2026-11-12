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
  file:    halton_seq.py

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

from typing import overload
import numpy as np

@overload
def halton_sequence(index:int, base:int) -> float: 
  ...

@overload
def halton_sequence(index:np.ndarray | list | tuple, base:int) -> np.ndarray: 
  ...

def _halton_sequence_scalar(index:int, base:int) -> float:
  """
  Compute the Halton sequence for a given index and base.

  :param int index:
    A non-negative integer representing the index in the Halton sequence.
  :param int base:
    A prime number greater than 1 representing the base for the Halton sequence.
  :returns:
    A float representing the Halton sequence value corresponding to the input index.
  :rtype:
    float
  """
  f = 1.0
  r = 0.0
  while (index > 0):
    f = f / float(base)
    r = r + f * float(index % base)
    index = index // base
  return r

def _halton_sequence_vector(indices:np.ndarray, base:int) -> np.ndarray:
  """
  Compute the Halton sequence for a vector of indices and a given base.
  
  :param numpy.ndarray indices:
    A numpy array of non-negative integers representing the indices in the Halton sequence.
  :param int base:
    A prime number greater than 1 representing the base for the Halton sequence.
  :returns:
    A numpy array of floats representing the Halton sequence values corresponding to the input indices.
  :rtype:
    numpy.ndarray
  :raises ValueError:
    If any index is not an integer, if it is negative or if the base is less than or equal to 1.
  """
  f = np.ones_like(indices, dtype=np.float64)
  r = np.zeros_like(indices, dtype=np.float64)
  
  while np.any(indices > 0):
    f /= float(base)
    r += f * (indices % base)
    indices //= base
  
  return r

def halton_sequence(index:int|np.ndarray, base:int) -> float|np.ndarray:
  """
  Compute the Halton sequence for a given index or an array of indices and a specified base.
  Automatically dispatches to the appropriate function (:py:func:`halton_sequence_scalar <lagrbfpy.halton_sequence_scalar>` 
  or :py:func:`halton_sequence_vector <lagrbfpy.halton_sequence_vector>`) based on the type of ``index``.
  
  :param int|numpy.ndarray index:
    An integer or a numpy array of non-negative integers representing the indices in the Halton sequence.
  :param int base:
    A prime number greater than 1 representing the base for the Halton sequence.
  :returns:
    A float or a numpy array of floats representing the Halton sequence values corresponding 
    to the input index/indices.
  :rtype:
    float or numpy.ndarray
  """
  if base <= 1:
    raise ValueError("base must be greater than 1.")
  
  if isinstance(index, (int, np.integer)):
    return _halton_sequence_scalar(index, base)
  elif isinstance(index, (np.ndarray, list, tuple)):
    # Convert the input to a numpy array of integers
    index = np.asarray(index, dtype=np.int64)
    if not np.issubdtype(index.dtype, np.integer):
      raise ValueError("index must be an array of integers.")
    if np.any(index < 0):
      raise ValueError("index must be non-negative.")
    return _halton_sequence_vector(index, base)
  else:
    raise TypeError(f"Index must be an integer or a numpy array of integers. Found index = {index} of type {type(index)}")

def test():
  pset = np.arange(15, dtype=np.int32)
  base = 3
  for i in pset:
    h = halton_sequence(i, base)
    print(f"H({i}, {base}) = {h}")
  
  hh = halton_sequence(pset, 3)
  print(hh)
  return

if __name__ == "__main__":
  test()