import os
import numpy as np
import pyvista as pvs
import lagrbfpy as lf

from petsc4py import PETSc

"""
run with
mpirun -np N python python_utils/scripts/interpolation_velocity.py

scaled by the following factors:
- length: 2.91e3 km
- time:   1.0 Myr
A and B chosen so the max velocity is ~3.5 cm/yr
"""

def interpolation(
    comm:PETSc.Comm,
    root:str,
    swarmO:np.ndarray, 
    swarmL:np.ndarray, 
    npoints:np.int32,
    velocity_data:lf.FlowData,
    write_vtp:bool=True
  ) -> None:
  options = PETSc.Options()
  rank    = comm.getRank()
  # Number of data points => resolution of the data
  ppd        = np.int32(361)
  npdata     = np.int32(ppd**3)
  # create swarms
  # note that we use different bases for the data and interpolation swarms to avoid having the same points in both swarms
  swarm_data = lf.PointSwarm(
    comm=comm,
    O=swarmO, L=swarmL, n=npdata,
    point_type="halton",
    coordinate_type="spherical",
    bases=[7,11,13] 
  )
  swarm_interp = lf.PointSwarm(
    comm=comm,
    O=swarmO, L=swarmL, n=npoints,
    point_type="halton",
    coordinate_type="spherical",
  )
  nfields = 3
  # set data values
  swarm_data.point_data["velocity"] = lf.toroidal_poloidal_flow(swarm_data.points, velocity_data)

  # points
  points = lf.PointContainer(coords=swarm_interp.points)
  # data
  data = lf.PointContainer(
    coords=swarm_data.points,
    values=swarm_data.point_data["velocity"]
  )
  # create points to hold the output
  output_coords = np.zeros((npoints, 3), dtype=np.float64)
  output_values = np.zeros((npoints, 3), dtype=np.float64)
  output_pidx   = np.zeros((npoints,),   dtype=np.int64)
  output = lf.PointContainer(
    coords=output_coords,
    values=output_values,
    pidx=output_pidx
  )
  kernel_type = {
    "wendland_c0": 1,
    "wendland_c2": 2,
    "bump": 3,
    "gaussian": 6
  }
  kernel_names = {
    1: "wendland_c0",
    2: "wendland_c2",
    3: "bump",
    6: "gaussian"
  }
  kernel = kernel_type["wendland_c2"]
  ctx = lf.LagRBFInterpolator(
    options = options,
    dim = 3,
    minCoords = np.array([ -2.5, -2.5, -2.5], dtype=np.float64),
    maxCoords = np.array([  2.5,  2.5,  2.5], dtype=np.float64),
    npartitions = np.array([96, 96, 96], dtype=np.int32),
    nfields = nfields,
    kernel = kernel,
    overlap = 1.1,
    shape_fac = 0.25,
  )
  ctx.interpolate(
    data,
    points,
    output
  )

  if rank == 0:
    swarm_out = pvs.PolyData(np.float64(output.coords))
    swarm_out.point_data["velocity"] = output.values
    swarm_out.point_data["pidx"] = output.pidx
    swarm_out.point_data["velocity_exact"] = lf.toroidal_poloidal_flow(swarm_out.points, velocity_data)
    swarm_out.point_data["error"] = np.linalg.norm(swarm_out.point_data["velocity"] - swarm_out.point_data["velocity_exact"], axis=1) / np.max(np.linalg.norm(swarm_out.point_data["velocity_exact"], axis=1))
    # compute global error metrics
    l2_error   = np.sqrt(np.mean(swarm_out.point_data["error"]**2))
    linf_error = np.max(swarm_out.point_data["error"])
    print(f"Error metrics for {npdata} data points and kernel {kernel_names[kernel]}:")
    print(f"  L2 error: {l2_error:.3e}")
    print(f"  Linf error: {linf_error:.3e}")
    if write_vtp:
      if not os.path.exists(root):
        os.makedirs(root)
      swarm_out.save(os.path.join(root, f"interpolation_velocity_{kernel}_{ppd}.vtp"))

  del ctx
  return

def main(root, degree_t, degree_p, order_t, order_p, radial_dt, radial_dp, A, B):
  comm = PETSc.COMM_WORLD
  outdirname = f"Ht{degree_t}_{order_t}-Hp{degree_p}_{order_p}-rt{radial_dt}-rp{radial_dp}-A{A:.1e}-B{B:.1e}"
  if degree_t == 0:
    mtype = "Poloidal"
  elif degree_p == 0:
    mtype = "Toroidal"
  else:
    mtype = "Mixed"
  root = os.path.join(root, "interpolation", mtype, outdirname)

  r0 = 1.2 # inner radius
  r1 = 2.2 # outer radius

  swarmO  = np.array([r0, 0, 0], dtype=np.float64)
  swarmL  = np.array([r1, 2*np.pi, 2*np.pi], dtype=np.float64)
  npoints = np.int32(32768) #32**3

  # setup velocity data structure
  velocity_data = lf.FlowData(
    degree_t=np.int32(degree_t),
    degree_p=np.int32(degree_p),
    order_t=np.int32(order_t),
    order_p=np.int32(order_p),
    radial_dt=np.int32(radial_dt),
    radial_dp=np.int32(radial_dp),
    A=np.float64(A),
    B=np.float64(B),
    r0=np.float64(r0),
    r1=np.float64(r1)
  )

  interpolation(comm, root, swarmO, swarmL, npoints, velocity_data, write_vtp=True)

  return

if __name__ == "__main__":
  root = os.path.join(
    os.environ["SCRATCH"], 
    "LagRBF_benchmark"
  )
  models = [
    # Toroidal models
    (1,  0, 1, 0.5*1e-2,   0,  0, 0,  0.0), # T 0, 0,1 A 1e-2,   P 0,0,0 B 0.0
    (2,  0, 2, 0.5*1e-3,   0,  0, 0,  0.0), # T 0, 0,2 A 1e-3,   P 0,0,0 B 0.0
    (3, -2, 3, 0.5*1.5e-4, 0,  0, 0,  0.0), # T 0,-2,3 A 1.5e-4, P 0,0,0 B 0.0
    (6, -3, 6, 0.5*5.0e-7, 0,  0, 0,  0.0), # T 0,-3,6 A 5.0e-7, P 0,0,0 B 0.0
    # Poloidal models
    (0,  0, 0, 0.0,    1,  0, 1, 0.5*4e-2), # T 0, 0,0 A 0.0,    P 1,0,1 B 4e-2
    (0,  0, 0, 0.0,    2, -1, 2, 0.5*4e-3), # T 0, 0,0 A 0.0,    P 2,-1,2 B 4e-3
    (0,  0, 0, 0.0,    3,  1, 3, 0.5*6e-4), # T 0, 0,0 A 0.0,    P 3,1,3 B 6e-4
    (0,  0, 0, 0.0,    6, -3, 6, 0.5*4e-6), # T 0, 0,0 A 0.0,    P 6,-3,6 B 4e-6
    # Mixed models
    (1,  0, 1, 0.5*8e-3,   1,  0, 1, 0.5*2.5e-2), # T 0, 0,1 A 8e-3,   P 1,0,1 B 2.5e-2
    (2,  0, 2, 0.5*8e-4,   2, -1, 2, 0.5*2.5e-3), # T 0, 0,2 A 8e-4,   P 2,-1,2 B 2.5e-3
    (3, -2, 3, 0.5*9e-5,   3,  1, 3, 0.5*4e-4), # T 0,-2,3 A 9e-5,   P 3,1,3 B 4e-4
    (6, -3, 6, 0.5*3e-7,   6, -3, 6, 0.5*4e-6), # T 0,-3,6 A 3e-7,   P 6,-3,6 B 4e-6
  ]

  run_model = [models[10]]

  for model in run_model:
    degree_t, order_t, radial_dt, A, degree_p, order_p, radial_dp, B = model
    lf.print("Running advection benchmark with parameters:")
    lf.print(f"  degree_t: {degree_t}, order_t: {order_t}, radial_dt: {radial_dt}")
    lf.print(f"  degree_p: {degree_p}, order_p: {order_p}, radial_dp: {radial_dp}")
    lf.print(f"  A: {A}, B: {B}")
    # run the model, set interpolate to True to run with interpolation and False to run with analytical velocity function
    main(root, degree_t, degree_p, order_t, order_p, radial_dt, radial_dp, A, B)