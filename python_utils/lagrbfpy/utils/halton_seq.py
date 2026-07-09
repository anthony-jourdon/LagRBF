import numpy as np

def halton_sequence_scalar(index:int, base:int) -> float:
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
  
  Parameters:
  - indices: A numpy array of non-negative integers representing the indices in the Halton sequence.
  - base: An integer greater than 1 representing the base for the Halton sequence.
  
  Returns:
  - A numpy array of floats representing the Halton sequence values corresponding to the input indices.
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
  
  Parameters:
  - index: An integer or a numpy array of non-negative integers representing the indices in the Halton sequence.
  - base: An integer greater than 1 representing the base for the Halton sequence.
  
  Returns:
  - A float or a numpy array of floats representing the Halton sequence values corresponding to the input index/indices.
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