import os
import numpy as np
import lagrbfpy as lf
import pyvista as pvs

# First 30 step:
# divergence threshold: 1e5
# e2 threshold: 1e4
# After 30 steps:
# divergence threshold: 1e4
# e2 threshold: 1e3

def copy_swarm(swarm:pvs.PolyData) -> pvs.PolyData:
  new_swarm = pvs.PolyData(swarm.points)
  for f in swarm.point_data:
    new_swarm.point_data[f] = np.copy(swarm.point_data[f])
  return new_swarm

def map_tect_system(geol_swarm:pvs.PolyData, swarm:pvs.PolyData, order:np.ndarray, pidx:np.ndarray, time:float):
  systems = {}
  if time < 1.05e-04: thresh = 1e5
  else: thresh = 1e4
  print(geol_swarm.number_of_points, swarm.number_of_points, pidx.shape[0], order.shape[0])

  divergence = swarm.point_data['divergence'][order]
  systems["extension"] = np.zeros(geol_swarm.number_of_points, dtype=bool)
  systems["extension"][pidx-1] = divergence > thresh
  systems["compression"] = np.zeros(geol_swarm.number_of_points, dtype=bool)
  systems["compression"][pidx-1] = divergence < -thresh
  cnt = 0
  for s in systems:
    geol_swarm.point_data['system'][systems[s]] = cnt + 1
    geol_swarm.point_data['system_age'][systems[s]] = time
    cnt += 1
  return

def map_deformation(geol_swarm:pvs.PolyData, swarm:pvs.PolyData, order:np.ndarray, pidx:np.ndarray, time:float):
  if time < 1.05e-04: thresh = 1e4
  else: thresh = 1e3
  e2 = np.zeros(geol_swarm.number_of_points, dtype=np.float64)
  e2[pidx-1] = swarm.point_data['e2'][order]
  active = np.zeros(geol_swarm.number_of_points, dtype=bool)
  active[pidx-1] = e2[pidx-1] > thresh
  # reset active deformation to 0 for points that are not active
  geol_swarm.point_data['deformation_active'] = np.zeros(geol_swarm.number_of_points, dtype=np.float64)
  geol_swarm.point_data['deformation_active'][active] = e2[active]
  geol_swarm.point_data["deformation_age"][active] = time
  return

def set_age(geol_swarm:pvs.PolyData, prev_geol_swarm:pvs.PolyData, time:float, field:str):
  geol_swarm.point_data[field+"_age"][geol_swarm.point_data[field] != prev_geol_swarm.point_data[field]] = time
  return

def map_step(geol_swarm:pvs.PolyData, prev_geol_swarm:pvs.PolyData, step_fname:str, time:float) -> None:
  swarm:pvs.PolyData = pvs.read(step_fname)
  order = np.argsort(swarm.point_data['pidx'])
  pidx  = swarm.point_data['pidx'][order]

  # Update geol_swarm points and data based on pidx matching
  # pidx is 1-based, so we need to adjust for 0-based indexing in Python
  geol_swarm.points[pidx-1, :] = swarm.points[order, :]

  map_tect_system(geol_swarm, swarm, order, pidx, time)
  map_deformation(geol_swarm, swarm, order, pidx, time)
  #set_age(geol_swarm:pvs.PolyData, prev_geol_swarm:pvs.PolyData, time:float, field:str)
  return

def initialize_geolmap(init_fname:str) -> pvs.PolyData:
  # load the initial swarm containing point identifier and geological region information
  init_swarm:pvs.PolyData = pvs.read(init_fname)
  # create a new swarm for geological mapping from the initial swarm
  geolswarm:pvs.PolyData = pvs.PolyData(init_swarm.points)
  # copy the unique point identifier from the initial swarm to the geol swarm
  geolswarm.point_data['pidx'] = np.copy(init_swarm.point_data['pidx'])

  # Initialize the type of rock (protolithe) to 0 (mantle) for all points
  geolswarm.point_data["protolithe"] = np.zeros(geolswarm.number_of_points, dtype=np.int32)
  geolswarm.point_data["system"] = np.zeros(geolswarm.number_of_points, dtype=np.int32)
  geolswarm.point_data["system_age"] = np.full(geolswarm.number_of_points, np.nan, dtype=np.float64)
  geolswarm.point_data["deformation_active"] = np.zeros(geolswarm.number_of_points, dtype=np.float64)
  geolswarm.point_data["deformation_age"] = np.full(geolswarm.number_of_points, np.nan, dtype=np.float64)
  
  regions = {}
  cnt = 0
  for f in init_swarm.point_data:
    if f == "pidx": continue
    regions[f] = init_swarm.point_data[f] > 0.1
    geolswarm.point_data["protolithe"][regions[f]] = cnt + 1
    cnt += 1
  
  return geolswarm

def create_map():
  root       = os.path.join(os.environ["DATA"],"RBF-Continents")
  pvd_geol   = "timeseries-geolmap.pvd"
  pvd_global = os.path.join(root, "timeseries-geolmap-total.pvd")
  pvd_llsvp  = os.path.join("/data","pandora","llsvp-pandora-vtu","surface-yy","timeseries_llsvp-surface-total.pvd")

  if not os.path.exists(pvd_global):
    time_global = np.float64(0.0)
  else:
    global_ = lf.timeseries_compare(pvd_global,pvd_llsvp)
    if global_[1] is None:
      print(f"Nothing to do.")
      exit(0)
    else:
      time_global = np.float64(global_[1]["time"][-1])
    

  models_dirs = [
    "Run-llsvp",
    "Run-llsvp2",
    "Run-llsvp3",
    "LLSVP-Run3",
    "LLSVP-Run4",
    "LLSVP-Run5",
    "LLSVP-Run6",
    "LLSVP-Run7",
    "LLSVP-Run8",
    "LLSVP-Run9"
  ]
  model_to_process = [
    "Run-llsvp",
    "Run-llsvp2",
    "Run-llsvp3",
    "LLSVP-Run3",
    "LLSVP-Run4",
    "LLSVP-Run5",
  ]
  timeseries_out        = {}
  timeseries_out_global = {}
  
  for model_dir in model_to_process:
    d = models_dirs.index(model_dir)
    print(f"Processing model: {model_dir}, index: {d}")

    iou = lf.IOutils(
      basedir=root,
      model_dir=model_dir,
      output_dir=os.path.join(root, model_dir, "geolmap"),
      pvd_out=pvd_geol
    )

    pvd_in  = os.path.join(iou.basedir, iou.mdir, "geodyn", "timeseries-swarm_interp.pvd")
    pvd_out = os.path.join(iou.outdir, iou.pvd)

    if not os.path.exists(pvd_out):
      print(f"File {pvd_out} not found, creating new geological swarm.")
      # load data timeseries
      timeseries = lf.timeseries_process(pvd_in)
      if model_dir == "Run-llsvp":
        # create the geological mapping swarm
        geol_swarm:pvs.PolyData = initialize_geolmap(os.path.join(iou.basedir, "ContinentsInit", "step000000-fields-swarm_interp.vtp"))
      else:
        prev_pvd = os.path.join(root, models_dirs[d-1], "geolmap", pvd_geol)
        if not os.path.exists(prev_pvd):
          raise FileNotFoundError(f"Previous geological mapping pvd file not found: {prev_pvd}. Cannot continue mapping for model {model_dir}.")
        prev_timeseries = lf.timeseries_process(prev_pvd)
        geol_swarm:pvs.PolyData = pvs.read(os.path.join(root, models_dirs[d-1], "geolmap", prev_timeseries['fname'][-1]))
    else:
      print(f"Found existing geological mapping pvd file: {pvd_out}, comparing with input pvd file: {pvd_in}")
      timeseries, geol_timeseries, _ = lf.timeseries_compare(pvd_in, pvd_out)
      if geol_timeseries is None:
        # Nothing to do for this model, continue to next one
        print(f"Model {model_dir} already completed, moving to next model.")
        continue
      # load last mapped geol swarm
      geol_swarm:pvs.PolyData = pvs.read(os.path.join(iou.outdir, geol_timeseries['fname'][-1]))
      step = int(geol_timeseries["step"][-1])
      print(f"Last mapped step: {step}. Continuing mapping from this step.")
    
    # copy swarm for age tracking
    previous_geol_swarm = copy_swarm(geol_swarm)

    for i in range(0, len(timeseries['step'])):
      step = timeseries['step'][i]
      time_local = timeseries['time'][i]

      if i > 0:
        time_global += np.float64(timeseries['time'][i]) - np.float64(timeseries['time'][i-1])

      print(f"Processing step {step} at time local {float(time_local):1.4e}, time global {time_global:1.4e}")
      step_fname = os.path.join(iou.basedir, iou.mdir, "geodyn", timeseries['fname'][i])

      map_step(geol_swarm, previous_geol_swarm, step_fname, time_global)
      previous_geol_swarm = copy_swarm(geol_swarm)

      geol_swarm.save(os.path.join(iou.outdir, f"step{str(step).zfill(6)}-geolmap.vtp"))
      timeseries_out["time"] = [str(time_local)]
      timeseries_out["step"] = [str(step).zfill(6)]
      lf.timeseries_write(os.path.join(iou.outdir, iou.pvd), timeseries_out, prefix="-geolmap", extension="vtp")
      timeseries_out_global["time"] = [str(time_global)]
      timeseries_out_global["step"] = [str(step).zfill(6)]
      timeseries_out_global["step_dir"] = [model_dir+"/geolmap/"]
      lf.timeseries_write(pvd_global, timeseries_out_global, prefix="-geolmap", extension="vtp")

  return

if __name__ == "__main__":
  create_map()