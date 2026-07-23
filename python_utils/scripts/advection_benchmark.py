from logging import root
import os
import numpy as np
import lagrbfpy as lf
import pyvista as pvs

from petsc4py import PETSc

"""
Note: numbers are scaled by the following factors:
- length: 2.91e3 km
- time:   1.0 Myr
A and B chosen so the max velocity is ~3.5 cm/yr
"""
def analytical(
    comm, 
    root:str,
    mtype:str,
    outdirname:str,
    swarmO:np.ndarray, 
    swarmL:np.ndarray, 
    npoints:np.int32,
    r0:float,
    r1:float,
    velocity_data:lf.FlowData,
  ) -> None:
  options = PETSc.Options()
  output_dir = os.path.join(root, "advection_analytical", mtype, outdirname)
  pvdfile    = "advection_analytical.pvd"
  rank = comm.getRank()
  swarm = lf.PointSwarm(
    comm=comm,
    O=swarmO, L=swarmL, n=npoints,
    point_type="halton",
    coordinate_type="spherical",
  )
    
  options["log_view"] = None
  advection = lf.AdvectionAnalytical(
    petsc_options=options,
    output_path=output_dir,
    dim=3,
    minCoords=np.array([ -2.5, -2.5, -2.5], dtype=np.float64),
    maxCoords=np.array([  2.5,  2.5,  2.5], dtype=np.float64),
    npartitions=np.array([3, 3, 3], dtype=np.int32),
    rmin=r0,
    rmax=r1,
    advection_cfl=0.1,
    time_start=0.0,
    time_end=10000.0,
    nsteps=101,
    pvdfilename=pvdfile
  )
  # initial state
  swarm.point_data["velocity"] = lf.toroidal_poloidal_flow(swarm.points, velocity_data)
  if rank == 0:
    swarm.save(os.path.join(output_dir, f"step{0:05d}.vtp"))
    timeseries = {
      "time": [f"{0.0:.6e}"], # store time as string
      "step": [f"{0:05d}"], # store step number as string
    }
    lf.timeseries_write(os.path.join(output_dir, pvdfile), timeseries=timeseries, prefix="", extension="vtp", erase=True)
  
  # set points to advect
  points = lf.PointContainer(coords=swarm.points)
  # create points to hold the output
  output_coords = np.zeros((npoints, 3), dtype=np.float64) # coordinates
  output_values = np.zeros((npoints, 3), dtype=np.float64) # velocity
  output_pidx   = np.zeros((npoints,),   dtype=np.int64)   # unique point index 1-based
  output = lf.PointContainer(coords=output_coords, values=output_values, pidx=output_pidx)

  # set advection function
  advection.set_advection_function(lf.Ctoroidal_poloidal_flow_batch, velocity_data)
  # advect
  advection.advect(points=points, output=output, comm=comm, prefix="")
  
  # free memory
  del advection
  return

def interpolation(
    comm, 
    root:str,
    mtype:str,
    outdirname:str,
    swarmO:np.ndarray, 
    swarmL:np.ndarray, 
    npoints:np.int32,
    r0:float,
    r1:float,
    velocity_data:lf.FlowData,
  ):
  options    = PETSc.Options()
  rank       = comm.getRank()
  output_dir = os.path.join(root, "advection_interpolation", mtype, outdirname)
  pvdfile    = "advection_interpolation.pvd"

  # create points
  swarm_data = lf.PointSwarm(
    comm=comm,
    O=swarmO, L=swarmL, n=npoints,
    point_type="halton",
    coordinate_type="spherical",
  )
  swarm_interp = lf.PointSwarm(
    comm=comm,
    O=swarmO, L=swarmL, n=npoints,
    point_type="halton",
    coordinate_type="spherical",
  )
  
  # create the advection object
  nfields = 3
  options["log_view"] = None
  advection = lf.AdvectionInterpolation(
    petsc_options=options,
    output_path=output_dir,
    dim=3,
    nfields=nfields,
    minCoords=np.array([ -2.5, -2.5, -2.5], dtype=np.float64),
    maxCoords=np.array([  2.5,  2.5,  2.5], dtype=np.float64),
    npartitions=np.array([12, 12, 12], dtype=np.int32),
    overlap=1.2,
    shape_factor=0.25,
    rmin=r0,
    rmax=r1,
    advection_cfl=0.1,
    time_start=0.0,
    time_end=10000.0,
    nsteps=101,
    pvdfilename=pvdfile
  )

  # set data values
  swarm_data.point_data["velocity"] = lf.toroidal_poloidal_flow(swarm_data.points, velocity_data)

  # points to advect
  points = lf.PointContainer(coords=swarm_interp.points)
  # create points to hold the output
  output_coords = np.zeros((npoints, 3), dtype=np.float64)
  output_values = np.zeros((npoints, 3), dtype=np.float64)
  output_pidx   = np.zeros((npoints,),   dtype=np.int64)
  output = lf.PointContainer(
    coords=output_coords,
    values=output_values,
    pidx=output_pidx
  )
  # callback function to get the data for interpolation during advection
  def get_data(step:np.int32, path:str, rank:int) -> lf.PointContainer:
    if rank == 0:
      # read the data from the vtp file
      fname      = os.path.join(path, f"step{step:05d}.vtp")
      swarm_data = pvs.read(fname)
    else:
      swarm_data = lf.PointSwarm()
      swarm_data.point_data["velocity"] = -1
    data = lf.PointContainer(
      coords=swarm_data.points,
      values=swarm_data.point_data["velocity"]
    )
    return data
  
  # advect
  advection.advect(
    get_data=get_data,
    data_args={"step": 0, "path": os.path.join(root, "advection_analytical", mtype, outdirname), "rank": rank}, 
    points=points, 
    output=output, 
    comm=comm, 
    fields={"velocity": 3},
    prefix=""
  )
  del advection
  return

def main(root:str, degree_t:int, degree_p:int, order_t:int, order_p:int, radial_dt:int, radial_dp:int, A:float, B:float) -> None:
  comm    = PETSc.COMM_WORLD
  outdirname = f"Ht{degree_t}_{order_t}-Hp{degree_p}_{order_p}-rt{radial_dt}-rp{radial_dp}-A{A:.1e}-B{B:.1e}"
  if degree_t == 0:
    mtype = "Poloidal"
  elif degree_p == 0:
    mtype = "Toroidal"
  else:
    mtype = "Mixed"

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

  analytical(   comm, root, mtype, outdirname, swarmO, swarmL, npoints, r0, r1, velocity_data)
  interpolation(comm, root, mtype, outdirname, swarmO, swarmL, npoints, r0, r1, velocity_data)
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

  for model in models:
    degree_t, order_t, radial_dt, A, degree_p, order_p, radial_dp, B = model
    lf.print("Running advection benchmark with parameters:")
    lf.print(f"  degree_t: {degree_t}, order_t: {order_t}, radial_dt: {radial_dt}")
    lf.print(f"  degree_p: {degree_p}, order_p: {order_p}, radial_dp: {radial_dp}")
    lf.print(f"  A: {A}, B: {B}")
    main(root, degree_t, degree_p, order_t, order_p, radial_dt, radial_dp, A, B)