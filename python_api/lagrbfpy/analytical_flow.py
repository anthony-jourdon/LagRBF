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
  file:    analytical_flow.py

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

import ctypes
import numpy as np
import sphericart as spc

from petsc4py import PETSc
from threading import Lock

from .interface import FunctionData, FunctionPointer

# Cache one SolidHarmonics context per degree
_SOLID_HARMONICS_CACHE: dict[int, spc.SolidHarmonics] = {}
_SOLID_HARMONICS_CACHE_LOCK = Lock()

def get_solid_harmonics_ctx(degree: int) -> spc.SolidHarmonics:
  ctx = _SOLID_HARMONICS_CACHE.get(degree)
  if ctx is not None:
    return ctx
  # Thread-safe lazy creation
  with _SOLID_HARMONICS_CACHE_LOCK:
    ctx = _SOLID_HARMONICS_CACHE.get(degree)
    if ctx is None:
      ctx = spc.SolidHarmonics(l_max=degree)
      _SOLID_HARMONICS_CACHE[degree] = ctx
  return ctx

def clear_solid_harmonics_cache() -> None:
  with _SOLID_HARMONICS_CACHE_LOCK:
    _SOLID_HARMONICS_CACHE.clear()

class HarmonicFlow:
  def __init__(self, degree:int, order:int, fac:float=1.0, radial_degree:int|None=None) -> None:
    self.degree  = degree
    self.order   = order
    self.fac     = fac
    self.rdegree = radial_degree if radial_degree is not None else degree
    self.ctx     = get_solid_harmonics_ctx(degree)

    self.r = None
    return

  def evaluate_harmonics(self, coords:np.ndarray) -> np.ndarray|None:
    if coords.shape[0] == 0:
      return None
    # real spherical harmonics and their derivatives in cartesian coordinates
    _, dH = self.ctx.compute_with_gradients(coords)
    return dH[:, :, self.degree*(self.degree+1)+self.order]
  
  def radial_function(self, r:np.ndarray) -> np.ndarray:
    raise NotImplementedError("radial_function must be implemented in subclasses.")
  
  def compute_flow(self, coords:np.ndarray) -> np.ndarray:
    raise NotImplementedError("compute_flow must be implemented in subclasses.")

class ToroidalFlow(HarmonicFlow):
  def __init__(self, degree:int, order:int, fac:float=1.0, radial_degree:int|None=None) -> None:
    super().__init__(degree, order, fac, radial_degree)
    return
  
  def radial_function(self, r:np.ndarray) -> np.ndarray:
    return r**self.rdegree
  
  def compute_flow(self, coords:np.ndarray, r:np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
      return np.empty((0, 3), dtype=np.float64)
    
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    f  = self.radial_function(r)
    dH = self.evaluate_harmonics(coords)

    # compute the flow
    # uT = A * f * (dH x coords)
    uT = np.zeros_like(coords)
    uT[:,0] = f * (dH[:,1]*z - dH[:,2]*y)
    uT[:,1] = f * (dH[:,2]*x - dH[:,0]*z)
    uT[:,2] = f * (dH[:,0]*y - dH[:,1]*x)
    uT *= self.fac

    return uT
     
class PoloidalFlow(HarmonicFlow):
  def __init__(
      self, 
      degree:int,
      order:int, 
      r0:float=0.0, 
      r1:float=1.0, 
      n:int=2, 
      m:int=2,
      radial_degree:int|None=None
    ) -> None:
    super().__init__(degree, order, radial_degree=radial_degree)
    self.r0 = r0
    self.r1 = r1
    self.n  = n if n >= 2 else 2
    self.m  = m if m >= 2 else 2
    return
  
  def radial_function(self, r:np.ndarray) -> np.ndarray:
    degree = self.rdegree
    return r**(degree+1) * (r - self.r0)**self.n * (r - self.r1)**self.m
  
  def radial_function_derivative(self, r:np.ndarray) -> np.ndarray:
    degree = self.rdegree
    n  = self.n
    m  = self.m
    r0 = self.r0
    r1 = self.r1

    df  = (degree + 1) * r**degree * (r - r0)**n * (r - r1)**m
    df += r**(degree + 1) * ( 
      n * (r - r0)**(n - 1) * (r - r1)**m +
      m * (r - r0)**n * (r - r1)**(m - 1)
    )
    return df

  def compute_flow(self, coords:np.ndarray, r:np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
      return np.empty((0, 3), dtype=np.float64)
    
    degree = self.degree

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    f  = self.radial_function(r)
    df = self.radial_function_derivative(r)
    dH = self.evaluate_harmonics(coords)

    r_dot_dH = x*dH[:,0] + y*dH[:,1] + z*dH[:,2]
    # compute the flow
    # uP = B * ( df * ( r * dH - 1/r * coords * coords.dH ) + f * (degree + 1) * dH )
    uP = np.zeros_like(coords)
    uP[:,0] = df * ( r * dH[:,0] - x * r_dot_dH / r ) + f * (degree + 1) * dH[:,0]
    uP[:,1] = df * ( r * dH[:,1] - y * r_dot_dH / r ) + f * (degree + 1) * dH[:,1]
    uP[:,2] = df * ( r * dH[:,2] - z * r_dot_dH / r ) + f * (degree + 1) * dH[:,2]
    uP *= self.fac

    return uP

#class FlowData(ctypes.Structure):
class FlowData(FunctionData):
  _fields_ = [
    ("degree_t", ctypes.c_int32),
    ("degree_p", ctypes.c_int32),
    ("order_t", ctypes.c_int32),
    ("order_p", ctypes.c_int32),
    ("radial_dt", ctypes.c_int32),
    ("radial_dp", ctypes.c_int32),
    ("A", ctypes.c_double),
    ("B", ctypes.c_double),
    ("r0", ctypes.c_double),
    ("r1", ctypes.c_double)
  ]

def toroidal_poloidal_flow(coords:np.ndarray,data:FlowData) -> np.ndarray:
  if coords.shape[0] == 0:
    return -1
  
  x = coords[:, 0]
  y = coords[:, 1]
  z = coords[:, 2]
  r = np.sqrt(x**2 + y**2 + z**2)

  # Toroidal flow
  toroidal_flow = ToroidalFlow(data.degree_t, data.order_t, data.A, radial_degree=data.radial_dt)
  uT = toroidal_flow.compute_flow(coords, r)

  # Poloidal flow
  poloidal_flow = PoloidalFlow(data.degree_p, data.order_p, data.r0, data.r1, n=2, m=2, radial_degree=data.radial_dp)
  poloidal_flow.fac = data.B
  uP = poloidal_flow.compute_flow(coords, r)

  # Total flow
  u_total = uT + uP

  return u_total

@ctypes.CFUNCTYPE(
    ctypes.c_int, 
    ctypes.POINTER(
      np.ctypeslib.as_ctypes_type(
        np.dtype(PETSc.RealType)
      )
    ), 
    ctypes.POINTER(
      np.ctypeslib.as_ctypes_type(
        np.dtype(PETSc.RealType)
      )
    ), 
    ctypes.c_void_p
  )
def Ctoroidal_poloidal_flow(coords, velocity, data_ptr):
  data = ctypes.cast(
    data_ptr,
    ctypes.POINTER(FlowData)
  ).contents
  
  X = np.ctypeslib.as_array(coords, shape=(1,3))
  u_total = toroidal_poloidal_flow(X, data)
  velocity[0] = u_total[0, 0]
  velocity[1] = u_total[0, 1]
  velocity[2] = u_total[0, 2]
  return 0

@ctypes.CFUNCTYPE(
    ctypes.c_int,
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # npoints
    np.ctypeslib.as_ctypes_type(np.dtype(PETSc.IntType)), # dim
    ctypes.POINTER(
      np.ctypeslib.as_ctypes_type(
        np.dtype(PETSc.RealType)
      )
    ),
    ctypes.POINTER(
      np.ctypeslib.as_ctypes_type(
        np.dtype(PETSc.RealType)
      )
    ),
    ctypes.c_void_p
  )
def Ctoroidal_poloidal_flow_batch(npoints, dim, coords, velocity, data_ptr):
  data = ctypes.cast(
    data_ptr,
    ctypes.POINTER(FlowData)
  ).contents

  X = np.ctypeslib.as_array(coords, shape=(npoints, dim))
  V = np.ctypeslib.as_array(velocity, shape=(npoints, dim))
  V[:, :] = toroidal_poloidal_flow(X, data)
  return 0