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
  file:    timeseries.py

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

import re
import os
from pathlib import Path

def timeseries_process(pvdfname:str|Path, start_line:int|None=None) -> dict:
  """
  Reads a .pvd file and extracts the time series information. 
  It uses the specific regular expression 
  
  .. code-block:: python

    r"(?:^(.*(\\d+)?)/)?([Aa-zZ]+0+)?(\\d+)(.*)"
  
  to extract the time, the step directory and the step number from the file name.

  :param str|pathlib.Path pvdfname: 
    pvd file name to process
  :param int|None start_line: 
    Line number to start processing from (default is None, which means to start from the beginning of the file)
  :returns: 
    Dictionary containing the time series information:

    - ``"time"``: list of time values as strings

    - ``"step_dir"``: list of step directories (e.g., ``"step1"``) or empty string if not present

    - ``"step"``: list of step numbers as strings

    - ``"fname"``: list of file names as strings

    - ``"line"``: list of line numbers in the .pvd file where the time series information was found

  :rtype: dict
  """
  if not isinstance(pvdfname, (str, Path)):
    raise ValueError("pvdfname must be a string or a Path object.")
  if not os.path.exists(pvdfname):
    raise FileNotFoundError(f"File {pvdfname} does not exist.")
  timeseries = {
    "time":     [], # store time as string
    "step_dir": [], # store step directory
    "step":     [], # store step number as string
    "fname":    [], # store file name
    "line":     []  # store line number in the .pvd file
  }
  with open(pvdfname,'r') as fp:
    if start_line is None:
      # skip the 3 first lines
      fp.readline()
      fp.readline()
      fp.readline()
      l = 3
    else:
      l = 0
      for _ in range(start_line):
        fp.readline()
        l += 1
    for line in fp:
      res = line.rsplit(sep='"')[1::2]
      if not res: continue
      time_str = res[0]
      file_str = res[1]
      timeseries["time"].append(time_str)
      timeseries["fname"].append(file_str)
      timeseries["line"].append(l)
      # (?:^(.*(\d+)?)/)?.*0(\d+)(.*)
      matchObj = re.match(r'(?:^(.*(\d+)?)/)?([Aa-zZ]+0+)?(\d+)(.*)',file_str)
      if matchObj:
        if matchObj.group(1):
          timeseries["step_dir"].append(matchObj.group(1))
        else:
          timeseries["step_dir"].append('')
        timeseries["step"].append(matchObj.group(4))
      l += 1
  return timeseries
  
def timeseries_compare(pvdfname1:str|Path, pvdfname2:str|Path) -> tuple[dict, dict|None, dict|None]:
    """
    Compare two timeseries from .pvd files. 
    If they have different number of lines, 
    it will process the one with more lines starting from the line where they differ, 
    and return the new timeseries along with the previous timeseries and the full timeseries. 
    If they have the same number of lines, it will simply return the timeseries 
    for the first file and None for the others.
    
    :param str|pathlib.Path pvdfname1: 
      pvd file name for the first timeseries
    :param str|pathlib.Path pvdfname2: 
      pvd file name for the second timeseries
    :returns: 
      A tuple containing the new timeseries (if they differ), 
      the previous timeseries, and the full timeseries.
    :rtype: tuple[dict, dict | None, dict | None]
    """
    timeseries1 = timeseries_process(pvdfname1)
    timeseries2 = timeseries_process(pvdfname2)
    nlines1 = len(timeseries1['line'])
    nlines2 = len(timeseries2['line'])
    if nlines1 != nlines2:
      if nlines1 < nlines2:
        restart_line = timeseries2['line'][nlines1]
        return timeseries_process(pvdfname2, restart_line), timeseries1, timeseries2
      else:
        restart_line = timeseries1['line'][nlines2]
        return timeseries_process(pvdfname1, restart_line), timeseries2, timeseries1
    else:
      print("Timeseries have the same number of lines.")
      return timeseries1, None, None

def timeseries_write_step(time:str, step:str, extension:str, prefix:str|None=None, stepdir:str="") -> str:
  """
  Write a single step entry for a .pvd file.
  The entry is formatted as follows:

  .. code-block:: xml

    <DataSet timestep="{time}" file="{stepdir}step{step}{prefix}.{extension}"/>

  where ``{time}`` is the time value, 
  ``{stepdir}`` is the step directory (if any), 
  ``{step}`` is the step number, 
  ``{prefix}`` is an optional prefix for the file name, 
  and ``{extension}`` is the file extension (e.g., ``"vtp"``).

  :param str time: Time value for the step
  :param str step: Step number for the step
  :param str extension: File extension for the step file (default is ``"vtp"``)
  :param str|None prefix: Optional prefix for the step file name (default is None)
  :param str stepdir: Optional step directory for the step file (default is empty string)
  :return: Formatted string for the step entry in the .pvd file
  :rtype: str
  """
  if prefix is None:
    s = f'  <DataSet timestep="{time}" file="{stepdir}step{step}.{extension}"/>\n'
  else:
    s = f'  <DataSet timestep="{time}" file="{stepdir}step{step}{prefix}.{extension}"/>\n'
  return s

def timeseries_write_new(timeseries:dict, prefix:str|None=None, extension:str="vtp") -> str:
  """
  Write a new .pvd file content for a given timeseries.

  :param dict timeseries: 
    Dictionary containing the time series information 
    (as returned by :py:func:`timeseries_process <lagrbfpy.timeseries_process>`)
  :param str|None prefix: 
    Optional prefix for the step file names (default is None)
  :param str extension: 
    File extension for the step files (default is ``"vtp"``)
  :return: Formatted string for the entire .pvd file content
  :rtype: str
  """
  nsteps = len(timeseries["time"])
  steps_dir = timeseries.get("step_dir", [""]*nsteps)
  s = "<?xml version=\"1.0\"?>\n"
  s += "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
  s += "<Collection>\n"
  for step in range(nsteps):
    time = timeseries["time"][step]
    step_num = timeseries["step"][step]
    step_dir = steps_dir[step]
    s += timeseries_write_step(time, step_num, extension, prefix, step_dir)
  s += "</Collection>\n"
  s += "</VTKFile>"
  return s
  
def timeseries_append(content:str, timeseries:dict, prefix:str|None=None, extension:str="vtp") -> str:
  """
  Append new steps to an existing .pvd file content for a given timeseries.
  The function takes the existing content of the .pvd file as a list of lines,
  and appends new step entries for the given timeseries until it reaches the closing ``</Collection>`` tag.

  :param list content: 
    List of lines representing the existing content of the .pvd file
  :param dict timeseries: 
    Dictionary containing the time series information 
    (as returned by :py:func:`timeseries_process <lagrbfpy.timeseries_process>`)
  :param str|None prefix: 
    Optional prefix for the step file names (default is None)
  :param str extension: 
    File extension for the step files (default is ``"vtp"``)
  :return: Formatted string for the entire .pvd file content with the new steps appended
  :rtype: str
  """
  nsteps = len(timeseries["time"])
  steps_dir = timeseries.get("step_dir", [""]*nsteps)
  s = ""
  for line in content:
    if '</Collection>' in line:
      break
    s += line
  for step in range(nsteps):
    time = timeseries["time"][step]
    step_num = timeseries["step"][step]
    step_dir = steps_dir[step]
    s += timeseries_write_step(time, step_num, extension, prefix, step_dir)
  s += '  </Collection>\n'
  s += '</VTKFile>'
  return s
  
def timeseries_write(fname:str, timeseries:dict, prefix:str|None=None, extension:str="vtp", erase:bool=False) -> None:
  """
  Write or append a timeseries to a .pvd file.
  If the file already exists and ``erase`` is ``False``, 
  it will append the new timeseries to the existing file content calling 
  :py:func:`timeseries_append <lagrbfpy.timeseries_append>`. 
  Otherwise, it will create a new file with the given timeseries as the content 
  calling :py:func:`timeseries_write_new <lagrbfpy.timeseries_write_new>`.

  :param str fname: 
    File name for the .pvd file to write to
  :param dict timeseries: 
    Dictionary containing the time series information 
    (as returned by :py:func:`timeseries_process <lagrbfpy.timeseries_process>`)
  :param str|None prefix: 
    Optional prefix for the step file names (default is None)
  :param str extension: 
    File extension for the step files (default is ``"vtp"``)
  :param bool erase: 
    Whether to erase the existing file content if the file already exists (default is False)
  """
  if os.path.exists(fname) and not erase:
    with open(fname,'r') as f:
      content = f.readlines()
    s = timeseries_append(content, timeseries, prefix, extension)
  else:
    s = timeseries_write_new(timeseries, prefix, extension)
  with open(fname,'w') as f:
    f.write(s)
  return
