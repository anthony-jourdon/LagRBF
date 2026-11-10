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