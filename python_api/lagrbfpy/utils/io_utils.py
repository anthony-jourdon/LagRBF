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
  file:    io_utils.py

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

import os

class IOutils:
  def __init__(
    self,
    basedir:str,
    model_dir:str,
    output_dir:str,
    timeseries:dict[str, list]|None=None,
    step=0,
    prefix:str="",
    dm_name:str="",
    pvd_out:str|None=None
  ):
    
    self.basedir = basedir
    self.mdir    = model_dir
    self.outdir  = output_dir
    if not os.path.exists(self.outdir):
      os.makedirs(self.outdir)
      print(f"Created output directory: {self.outdir}")

    self.prefix  = prefix
    self.dm_name = dm_name
    
    self.step:int = step
    self.timeseries:dict[str, list]|None = timeseries
    self.pvd:str|None = pvd_out
    return
  
  def __str__(self) -> str:
    s  = f"class IOutils:\n"
    s += f"  basedir: {self.basedir}\n"
    s += f"  model_dir: {self.mdir}\n"
    s += f"  output_dir: {self.outdir}\n"
    s += f"  prefix: {self.prefix}\n"
    s += f"  dm_name: {self.dm_name}\n"
    s += f"  step: {self.step}\n"
    s += f"  pvd_out: {self.pvd}\n"
    s += f"  timeseries: {self.timeseries}\n"
    return s