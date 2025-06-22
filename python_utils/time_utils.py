import re

def process_timeseries(pvdfname:str) -> dict:
  print(f"Filtering temporal data from {pvdfname}")
  timeseries = {
    "time": [],
    "step_dir": [],
    "step": [],
    "fname": []
  }
  with open(pvdfname,'r') as fp:
    # skip the 3 first lines
    fp.readline()
    fp.readline()
    fp.readline()
    for line in fp:
      res = line.rsplit(sep='"')[1::2]
      if not res: continue
      time_str = res[0]
      file_str = res[1]
      timeseries["time"].append(time_str)
      timeseries["fname"].append(file_str)
      matchObj = re.match(r'(?:^(step\d+)/)?.*(\d{6})(.*)',file_str)
      if matchObj:
        if matchObj.group(1):
          timeseries["step_dir"].append(matchObj.group(1))
        else:
          timeseries["step_dir"].append('')
        timeseries["step"].append(matchObj.group(2))
  return timeseries

def write_timeseries(fname:str,timeseries:dict,prefix:str,extension:str):
  print(f"Writing timeseries to {fname}")
  nsteps = len(timeseries["time"])
  with open(fname,'w') as fp:
    fp.write('<?xml version="1.0"?>\n')
    fp.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    fp.write('<Collection>\n')
    for step in range(nsteps):
      fp.write('  <DataSet timestep="'+timeseries['time'][step]+'" file="step'+timeseries['step'][step]+'-'+prefix+'.'+extension+'"/>\n')
    fp.write('</Collection>\n')
    fp.write('</VTKFile>')
  return

def test():
  import os
  root = os.path.dirname(__file__)
  pvdfname = os.path.join(root, '../output/timeseries_X.pvd')
  if not os.path.exists(pvdfname):
    print(f"File {pvdfname} does not exist.")
    return

  timeseries = process_timeseries(pvdfname)
  print(timeseries)
  # Write a new timeseries
  write_timeseries(os.path.join(root,'../output/timeseries_test.pvd'),timeseries,'test','vtp')
  return

if __name__ == '__main__':
  test()