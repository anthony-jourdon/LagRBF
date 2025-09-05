import re

class XDMFDataFilter:
  def __init__(self, xdmfile:str):
    self.xdmfile = xdmfile
    self.timeseries = None
    self.fieldnames = None
    self.filenames  = None
    self.process_xdmf()
    return
  
  def __str__(self):
    s = "XDMFDataFilter class:\n"
    s += f"  XDMF File: {self.xdmfile}\n"
    s += f"  Timeseries:\n{self.timeseries}\n"
    s += f"  Field Names:\n{self.fieldnames}\n"
    s += f"  Filenames:\n{self.filenames}\n"
    return s

  def get_timeseries(self) -> list:
    if self.timeseries is None:
      raise ValueError("Timeseries not processed. Call process_xdmf() first.")
    return self.timeseries
  
  def get_fieldnames(self) -> list:
    if self.fieldnames is None:
      raise ValueError("Field names not processed. Call process_xdmf() first.")
    return self.fieldnames
  
  def get_filenames(self) -> list:
    if self.filenames is None:
      raise ValueError("Filenames not processed. Call process_xdmf() first.")
    return self.filenames

  def read_xdmf_lines(self) -> list:
    try:
      with open(self.xdmfile, 'r') as fp:
        lines = [line.rstrip() for line in fp]
      return lines
    except FileNotFoundError:
      raise FileNotFoundError(f"XDMF file {self.xdmfile} not found.")
  
  def time_filter(self, line:str) -> float|None:
    matchObj = re.match(r'.*<Time.*([0-9]{1,6}\.[0-9]+[Ee]?[\+\-]?[0-9]+).*', line)
    if matchObj:
      time = float(matchObj.group(1))
      return time
    return None
  
  def field_name_filter(self, line:str) -> str|None:
    matchObj = re.match(r'.*<Attribute.*Name=\"([a-zA-Z]+)\".*', line)
    if matchObj:
      field_name = matchObj.group(1)
      return field_name
    return None
  
  def file_name_filter(self, line:str) -> str|None:
    matchObj = re.match(r'\s+(\w+\.h5).*', line)
    if matchObj:
      file_name = matchObj.group(1)
      return file_name
    return None
  
  def process_xdmf(self) -> None:
    lines = self.read_xdmf_lines()
    self.timeseries = []
    self.fieldnames = []
    self.filenames  = []
    # Iterate through each line in the XDMF file
    for line in lines:
      # filter time values
      time = self.time_filter(line)
      if time is not None:
        self.timeseries.append(time)
      # filter field names
      fieldname = self.field_name_filter(line)
      if fieldname is not None:
        # ensure fieldname is unique
        if fieldname not in self.fieldnames:
          self.fieldnames.append(fieldname)
      # filter file names
      filename = self.file_name_filter(line)
      if filename is not None:
        # ensure filename is unique
        if filename not in self.filenames:
          self.filenames.append(filename)
    return