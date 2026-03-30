import os
import numpy as np
import lagrbfpy as lf
import pyvista as pvs

def map_tect_system(geol_swarm:pvs.PolyData, swarm:pvs.PolyData, order:np.ndarray, pidx:np.ndarray):
  systems = {}

  print(geol_swarm.number_of_points, swarm.number_of_points, pidx.shape[0], order.shape[0])

  divergence = swarm.point_data['divergence'][order]
  systems["extension"] = np.zeros(geol_swarm.number_of_points, dtype=bool)
  systems["extension"][pidx-1] = divergence > 1e5
  systems["compression"] = np.zeros(geol_swarm.number_of_points, dtype=bool)
  systems["compression"][pidx-1] = divergence < -1e5
  cnt = 0
  for s in systems:
    geol_swarm.point_data['system'][systems[s]] = cnt + 1
    cnt += 1
  return

def map_step(geol_swarm:pvs.PolyData, step_fname:str) -> None:
  swarm:pvs.PolyData = pvs.read(step_fname)
  order = np.argsort(swarm.point_data['pidx'])
  pidx  = swarm.point_data['pidx'][order]

  # Update geol_swarm points and data based on pidx matching
  # pidx is 1-based, so we need to adjust for 0-based indexing in Python
  geol_swarm.points[pidx-1, :] = swarm.points[order, :]

  map_tect_system(geol_swarm, swarm, order, pidx)
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
    
    for i in range(0, len(timeseries['step'])):
      step = timeseries['step'][i]
      time_local = timeseries['time'][i]

      if i > 0:
        time_global += np.float64(timeseries['time'][i]) - np.float64(timeseries['time'][i-1])

      print(f"Processing step {step} at time local {float(time_local):1.4e}, time global {time_global:1.4e}")
      step_fname = os.path.join(iou.basedir, iou.mdir, "geodyn", timeseries['fname'][i])
      map_step(geol_swarm, step_fname)
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