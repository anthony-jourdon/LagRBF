import re
import os
from pathlib import Path

class TimeSeries:
  def __init__(self) -> None:
    return

  def process(self, pvdfname:str, start_line:int|None=None) -> dict:
    timeseries = {
      "time": [],
      "step_dir": [],
      "step": [],
      "fname": [],
      "line": []
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
        matchObj = re.match(r'(?:^(step\d+)/)?.*(\d{6})(.*)',file_str)
        if matchObj:
          if matchObj.group(1):
            timeseries["step_dir"].append(matchObj.group(1))
          else:
            timeseries["step_dir"].append('')
          timeseries["step"].append(matchObj.group(2))
        l += 1
    return timeseries
  
  def compare(self, pvdfname1:str, pvdfname2:str) -> tuple[dict, dict|None, dict|None]:
    """
    Compare two timeseries from .pvd files. 
    If they have different number of lines, 
    it will process the one with more lines starting from the line where they differ, 
    and return the new timeseries along with the previous timeseries and the full timeseries. 
    If they have the same number of lines, it will simply return the timeseries for the first file and None for the others.
    
    :param self: Description
    :param pvdfname1: Description
    :type pvdfname1: str
    :param pvdfname2: Description
    :type pvdfname2: str
    :return: Description
    :rtype: tuple[dict, dict | None, dict | None]
    """
    timeseries1 = self.process(pvdfname1)
    timeseries2 = self.process(pvdfname2)
    nlines1 = len(timeseries1['line'])
    nlines2 = len(timeseries2['line'])
    if nlines1 != nlines2:
      if nlines1 < nlines2:
        restart_line = timeseries2['line'][nlines1]
        return self.process(pvdfname2, restart_line), timeseries1, timeseries2
      else:
        restart_line = timeseries1['line'][nlines2]
        return self.process(pvdfname1, restart_line), timeseries2, timeseries1
    else:
      print("Timeseries have the same number of lines.")
      return timeseries1, None, None

  def write_step(self, time:str, step:str, extension:str, prefix:str|None=None) -> str:
    if prefix is None:
      s = f'  <DataSet timestep="{time}" file="step{step}.{extension}"/>\n'
    else:
      s = f'  <DataSet timestep="{time}" file="step{step}-{prefix}.{extension}"/>\n'
    return s

  def write_new(self, timeseries:dict, prefix:str|None=None, extension:str="vtp") -> str:
    nsteps = len(timeseries["time"])
    s = "<?xml version=\"1.0\"?>\n"
    s += "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    s += "<Collection>\n"
    for step in range(nsteps):
      time = timeseries["time"][step]
      step_num = timeseries["step"][step]
      s += self.write_step(time, step_num, extension, prefix)
    s += "</Collection>\n"
    s += "</VTKFile>"
    return s
  
  def append(self, content:str, timeseries:dict, prefix:str|None=None, extension:str="vtp") -> str:
    nsteps = len(timeseries["time"])
    s = ""
    for line in content:
      if '</Collection>' in line:
        break
      s += line
    for step in range(nsteps):
      time = timeseries["time"][step]
      step_num = timeseries["step"][step]
      s += self.write_step(time, step_num, extension, prefix)
    s += '  </Collection>\n'
    s += '</VTKFile>'
    return s
  
  def write(self, fname:str, timeseries:dict, prefix:str|None=None, extension:str="vtp", erase:bool=False) -> None:
    if os.path.exists(fname) and not erase:
      with open(fname,'r') as f:
        content = f.readlines()
      s = self.append(content, timeseries, prefix, extension)
    else:
      s = self.write_new(timeseries, prefix, extension)
    with open(fname,'w') as f:
      f.write(s)
    return

def process_timeseries(pvdfname:str, start_line:int|None=None) -> dict:
  timeseries = {
    "time": [],
    "step_dir": [],
    "step": [],
    "fname": [],
    "line": []
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
      matchObj = re.match(r'(?:^(step\d+)/)?.*(\d{6})(.*)',file_str)
      if matchObj:
        if matchObj.group(1):
          timeseries["step_dir"].append(matchObj.group(1))
        else:
          timeseries["step_dir"].append('')
        timeseries["step"].append(matchObj.group(2))
      l += 1
  return timeseries

def write_timeseries(fname:str,timeseries:dict,prefix:str|None=None,extension:str="vtp") -> None:
  if not fname.endswith('.pvd'):
    fname += '.pvd'
  print(f"Writing timeseries to {fname}")
  nsteps = len(timeseries["time"])
  with open(fname,'w') as fp:
    fp.write('<?xml version="1.0"?>\n')
    fp.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    fp.write('<Collection>\n')
    for step in range(nsteps):
      if prefix is None:
        fp.write('  <DataSet timestep="'+timeseries['time'][step]+'" file="step'+timeseries['step'][step]+'.'+extension+'"/>\n')
      else:
        fp.write('  <DataSet timestep="'+timeseries['time'][step]+'" file="step'+timeseries['step'][step]+'-'+prefix+'.'+extension+'"/>\n')
    fp.write('</Collection>\n')
    fp.write('</VTKFile>')
  return

def append_timeseries(fname:str,timeseries:dict,prefix:str|None=None,extension:str="vts") -> None:
  with open(fname,'r+') as f:
    content = f.readlines()
    f.seek(0)
    # write up to Collection
    for line in content:
      if '  </Collection>' in line:
        break
      f.write(line)
      
    # append new entries
    for time_entry in timeseries:
      if prefix is None:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}.{extension}"/>\n')
      else:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}_{prefix}.{extension}"/>\n')
    # write closing tags
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')
  return

def test():
  import os
  root = os.path.dirname(__file__)
  pvdfname_X = os.path.join(root, '../../test_2d_weights/timeseries_X.pvd')
  pvdfname_S = os.path.join(root, '../../test_2d_weights/timeseries-swarm_interp.pvd')
  if not os.path.exists(pvdfname_X):
    print(f"File {pvdfname_X} does not exist.")
    return

  series = TimeSeries()

  timeseries_X = series.process(pvdfname_X)
  timeseries_S = series.process(pvdfname_S)
  timeseries_new, timeseries_prev, timeseries_full = series.compare(pvdfname_X, pvdfname_S)
  
  series.write(pvdfname_S, timeseries_X, prefix='fields-swarm_interp', extension='vtp', erase=True)

  return

if __name__ == '__main__':
  test()