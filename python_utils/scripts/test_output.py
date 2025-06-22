import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from time_utils import process_timeseries, write_timeseries
from petscvec_loader import PetscVecLoad
from write_vtp import write_vtp

def main():
  root = os.path.dirname(__file__)
  root = os.path.join(root, '../../output')
  timeseries = process_timeseries(os.path.join(root, 'timeseries_X.pvd'))
  nsteps = len(timeseries["time"])
  dms = ['swarm_interp']
  for step in range(nsteps):
    step_idx = timeseries["step"][step]
    step_dir = timeseries["step_dir"][step]
    for dm in dms:
      fname = [
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_coords.pbvec"),
        os.path.join(root, step_dir, f"step{step_idx}.{dm}_fields.pbvec")
      ]
      coor  = PetscVecLoad(fname[0])
      field = PetscVecLoad(fname[1])
      bs = field.shape[1]
      output_fields = {}
      for i in range(bs):
        output_fields[f'f{i}'] = field[:, i]
      write_vtp(coor, output_fields, os.path.join(root, f'step{step:06d}-fields.vtp'))
  write_timeseries(os.path.join(root, 'timeseries-fields.pvd'), timeseries, 'fields', 'vtp')
  return

if __name__ == '__main__':
  main()