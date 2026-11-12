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
  file:    __init__.py

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

from .timeseries import timeseries_process, timeseries_compare, timeseries_write_step, timeseries_write_new, timeseries_append, timeseries_write
from .write_vtp import write_vtp
from .io_utils import IOutils
from .halton_seq import halton_sequence
from .points_container import PointContainer, fill_datafields

__all__ = [
  'timeseries_process',
  'timeseries_compare',
  'timeseries_write_step',
  'timeseries_write_new',
  'timeseries_append',
  'timeseries_write',
  'write_vtp',
  'IOutils',
  'halton_sequence',
  'PointContainer',
  'fill_datafields'
]