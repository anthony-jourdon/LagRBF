import numpy as np

def halton_sequence_scalar(index:int, base:int) -> float:
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

def halton_sequence_vector(indices:np.ndarray, base:int) -> np.ndarray:
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
  if not np.issubdtype(indices.dtype, np.integer):
    raise ValueError("Indices must be an array of integers.")
  if np.any(indices < 0):
    raise ValueError("Indices must be non-negative.")
  if base <= 1:
    raise ValueError("Base must be greater than 1.")

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
  if isinstance(index, (int, np.integer)):
    return halton_sequence_scalar(index, base)
  elif isinstance(index, (np.ndarray, list, tuple)):
    index = np.asarray(index, dtype=np.int64)
    return halton_sequence_vector(index, base)
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