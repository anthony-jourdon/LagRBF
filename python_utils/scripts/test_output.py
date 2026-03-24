import os
import lagrbfpy as lf
import numpy as np

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

class Field:
  def __init__(self, name:str, iou:IOutils):
    self.name:str = name
    self.iou:IOutils = iou
    self.data:np.ndarray|None = None
    return
  
  def load(self) -> None:
    raise NotImplementedError("Field is an abstract class, cannot be loaded directly.")
  
  def get_values(self, *args) -> np.ndarray:
    if self.data is None:
      self.load()
    return self.data
  
  def reset(self):
    self.data = None
    return

class Real(Field):
  def __init__(self, name:str, iou:IOutils):
    super().__init__(name, iou)
    return

  def load(self) -> None:
    stepdir = f"step{self.iou.step}"
    fname = f"step{self.iou.step:06d}{self.iou.prefix}.{self.iou.dm_name}_{self.name}.pbvec"
    self.data = lf.PetscVecLoad(os.path.join(self.iou.basedir,self.iou.mdir,stepdir,fname))
    return

class Integer(Field):
  def __init__(self, name:str, iou:IOutils):
    super().__init__(name, iou)
    return

  def load(self) -> None:
    stepdir = f"step{self.iou.step}"
    fname = f"step{self.iou.step:06d}{self.iou.prefix}.{self.iou.dm_name}_{self.name}.pbis"
    self.data = lf.PetscISLoadFromFile(os.path.join(self.iou.basedir,self.iou.mdir,stepdir,fname))
    return

class DerivedField(Field):
  def __init__(self, name:str, iou:IOutils, field:Field, block_idx:int, blocksize:int=1):
    super().__init__(name, iou)
    self.field = field

    start_idx = block_idx
    end_idx   = block_idx + blocksize
    self.block_idx = slice(start_idx, end_idx)
    return
  
  def load(self) -> None:
    if self.field.data is None:
      self.field.load()
    return 

  def get_values(self, *args) -> np.ndarray:
    if self.field.data is None:
      self.load()
    return self.field.data[:, self.block_idx]

def fields_instances(iou:IOutils, fields_names:dict[str, int]) -> dict[str, Field]:
  fields = {}
  fields["coords"] = Real("DMSwarmPIC_coor", iou)
  fields["fields"] = Real("fields", iou)
  fields["pidx"]   = Integer("DMSwarm_pid", iou)
  
  block_idx = 0
  for f in fields_names:
    fields[f] = DerivedField(f, iou, fields["fields"], block_idx=block_idx, blocksize=fields_names[f])
    block_idx += fields_names[f]

  return fields

def extract_fields(fields_names:dict[str, int], fields_classes:dict[str, Field]):
  """
  we have 3 different files for the fields:

  1. fields.pbvec: contains all the fields except pidx
  2. DMSwarmPIC_coor.pbvec: contains the coordinates
  3. DMSwarm_pid.pbis: contains the particle index
  
  1. & 2. are loaded with PetscVecLoad, and 3. is loaded with PetscISLoadFromFile

  we rely on user fields input to name the fiel IN THE ORDER provided by the user
  """
  output_fields = {}
  try:
    output_fields['coords'] = fields_classes['coords'].get_values()
  except Exception as e:
    raise ValueError(f"Error loading coordinates field: {e}")
  try:
    output_fields['pidx']   = fields_classes['pidx'].get_values()
  except Exception as e:
    print(f"Error loading particle index field: {e}\nContinuing without particle index field.")
  for f in fields_names:
    output_fields[f] = fields_classes[f].get_values()

  for f in fields_classes:
    fields_classes[f].reset()
  return output_fields

def write_step(iou:IOutils, fields_classes:dict[str, Field], fields_names:dict[str, int]) -> dict[str, np.ndarray]:
  output_fields = extract_fields(fields_names, fields_classes)
  coords = output_fields.pop("coords")
  
  outfname = f"step{str(iou.step).zfill(6)}-fields-{iou.dm_name}.vtp"
  lf.write_vtp(coords, output_fields, os.path.join(iou.outdir, outfname))
  return

def process_steps(iou:IOutils, fields_classes:dict[str, Field], fields_names:dict[str, int]) -> None:
  s = 0
  for step in iou.timeseries['step']:
    iou.step = int(step)
    print(f"Processing step {step}...")
    write_step(iou, fields_classes, fields_names)
    iou.timeseries["step"][s] = f"{iou.step:06d}"
    s += 1
  
  _ = iou.timeseries.pop("step_dir")
  lf.timeseries_write(os.path.join(iou.outdir, iou.pvd), iou.timeseries, prefix="-fields-swarm_interp", extension="vtp", erase=False)
  return

def main():
  root    = os.path.join(os.environ["SCRATCH"],"RBF_pandora_continents")
  mdir    = "Continents-LLSVP-Run3"
  pvd_in  = "timeseries_X.pvd"
  pvd_out = "timeseries-swarm_interp.pvd"

  pvd_in_path  = os.path.join(root, mdir, pvd_in)
  pvd_out_path = os.path.join(root, mdir, pvd_out)

  if os.path.exists(pvd_out_path):
    print(f"Found existing output pvd file: {pvd_out_path}, comparing with input pvd file: {pvd_in_path}")
    tseries = lf.timeseries_compare(pvd_in_path, pvd_out_path)
    timeseries = tseries[0]
  else:
    print(f"No existing output pvd file found at {pvd_out_path}, processing input pvd file: {pvd_in_path}")
    timeseries = lf.timeseries_process(pvd_in_path)
  
  iou = IOutils(
    basedir=root,
    model_dir=mdir,
    output_dir=os.path.join(root, mdir),
    timeseries=timeseries,
    pvd_out=pvd_out,
    prefix="_forward",
    dm_name="swarm_interp"
  )
  # name of the fields and their size, must appear in the order they are stored in the fields.pbvec file
  fields_names = {
    "velocity": 3,
    "pressure": 1
  }
  fields_classes = fields_instances(iou, fields_names)
  process_steps(iou, fields_classes, fields_names)

  return

def test():
  array = np.arange(12).reshape((4,3))
  
  extract_idx = slice(0, 2)
  print(f"Original array:\n{array}\n")
  print(f"Extracted array with slice {extract_idx}:\n{array[:,extract_idx]}\n")

  return

if __name__ == '__main__':
  main()
  #test()