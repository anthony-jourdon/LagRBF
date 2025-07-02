import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lagrbfpy.time_utils import process_timeseries, write_timeseries
from lagrbfpy.petscvec_loader import PetscVecLoad, PetscISLoadFromFile
from lagrbfpy.write_vtp import write_vtp

def output_data(root:str):
  output_fields = {}
  fname = [
    os.path.join(root,"step0/step000000.swarm_data_coords.pbvec"),
    os.path.join(root,"step0/step000000.swarm_data_fields.pbvec")
  ]
  coor  = PetscVecLoad(fname[0])
  dim   = coor.shape[1]
  field = PetscVecLoad(fname[1])
  bs = field.shape[1]
  output_fields['velocity'] = field[:, :dim]
  for i in range(dim, bs):
    output_fields[f'f{i-dim}'] = field[:, i]
  write_vtp(coor, output_fields, os.path.join(root, f'step000000-fields-swarm_data.vtp'))
  return

def main():
  root = os.path.dirname(__file__)
  root = os.path.join(root, '../../output') #_spherical
  timeseries = process_timeseries(os.path.join(root, 'timeseries_X.pvd'))
  nsteps = len(timeseries["time"])
  #output_data(root)
  dms = ['swarm_interp']
  for step in range(nsteps):
    step_idx = timeseries["step"][step]
    step_dir = timeseries["step_dir"][step]
    for dm in dms:
      print(f"Processing step {int(step_idx)} for dm {dm}")
      output_fields = {}
      fname = [
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_coords.pbvec"),
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_fields.pbvec"),
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_pidx.pbis"),
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_v_adv.pbvec")
      ]
      coor  = PetscVecLoad(fname[0])
      dim   = coor.shape[1]
      field = PetscVecLoad(fname[1], order='F')  # Column-Major Order
      pidx  = PetscISLoadFromFile(fname[2])

      v_adv = PetscVecLoad(fname[3])
      for i in range(v_adv.shape[1]):
        output_fields[f'v_adv_{i}'] = v_adv[:, i]

      bs = field.shape[1]
      output_fields['velocity'] = field[:, :dim]
      for i in range(dim, bs):
        output_fields[f'f{i-dim}'] = field[:, i]
      output_fields['pidx'] = pidx
      write_vtp(coor, output_fields, os.path.join(root, f'step{step:06d}-fields-{dm}.vtp'))
  write_timeseries(os.path.join(root, 'timeseries-fields.pvd'), timeseries, f'fields-{dm}', 'vtp')
  return

if __name__ == '__main__':
  main()