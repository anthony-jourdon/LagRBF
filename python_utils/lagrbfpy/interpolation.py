import os
import numpy as np
import lagrbfpy as lrb

class LagRBFInterpolator:
  def __init__(self, data_coords:np.ndarray, data_values:np.ndarray, options_file:str|None=None, **kwargs) -> None:
    
    if data_coords.ndim == 1:
      s = f"data_coords should be a 2D array of shape (ndata, dim), found shape {data_coords.shape}"
      raise ValueError(s)
    
    self.data_coords = data_coords
    self.data_values = data_values
    self.dim = self.data_coords.shape[1]
    if data_values.ndim > 1:
      self.nfields = data_values.shape[1]
    else:
      self.nfields = 1
    self.O = kwargs.get("O", self.bbox_origin(kwargs.get("padding_factor", 0.1)))
    self.L = kwargs.get("L", self.bbox_end(kwargs.get("padding_factor", 0.1)))

    part = np.full(self.dim, 8, dtype=np.int32)
    self.partitions = kwargs.get("partitions", part)
    self.options_file = options_file
    self.log_view  = kwargs.get("log_view", False)
    self.kernel    = kwargs.get("kernel", 2) # default to Wendland C2 kernel
    self.overlap   = kwargs.get("overlap", 1.0) # default to 1.0
    self.shape_fac = kwargs.get("shape_fac", 1.0) # default to 1.0

    if self.options_file is None:
      options = self.generate_options()
      basedir = os.getcwd()
      self.options_file = os.path.join(basedir, "lagrbf_interpolate_py.opts")
      with open(self.options_file, "w") as f:
        f.write(options)
    return
  
  def bbox_origin(self, padding_factor:float=0.1) -> np.ndarray:
    min_coords = np.min(self.data_coords, axis=0)
    min_dx = padding_factor * min_coords
    O = min_coords + min_dx
    return O
  
  def bbox_end(self, padding_factor:float=0.1) -> np.ndarray:
    max_coords = np.max(self.data_coords, axis=0)
    max_dx = padding_factor * max_coords
    L = max_coords + max_dx
    return L
  
  def generate_options(self) -> str:
    s  = f"-pu_mesh_dim {self.dim} # spatial dimension\n"
    s += f"-pu_mesh_n {self.partitions[0]}"
    for i in range(1, self.dim):
      s += f",{self.partitions[i]}"
    s += " # number of partitions in each direction\n"
    s += f"-pu_mesh_O {self.O[0]}"
    for i in range(1, self.dim):
      s += f",{self.O[i]}"
    s += " # mesh origin\n"
    s += f"-pu_mesh_L {self.L[0]}"
    for i in range(1, self.dim):
      s += f",{self.L[i]}"
    s += " # mesh end\n"
    s += f"-pu_mesh_overlap {self.overlap} # overlap factor between partitions\n"
    s += f"-pu_mesh_shape_factor {self.shape_fac} # shape factor for the RBF kernel\n"
    s += f"-n_interpolation_fields {self.nfields} # number of fields to interpolate\n"
    s += f"-kernel_type {self.kernel} # kernel type\n"
    s += "-solve_method 2\n"
    s += "-partition_pc_type cholesky\n"
    s += "-partition_ksp_type preonly\n"
    s += "-partition_ksp_rtol 1e-8\n"
    s += "-partition_ksp_atol 1e-8\n"
    s += "-partition_ksp_reuse_preconditioner true\n"
    s += "-partition_pc_factor_shift_type positive_definite\n"
    if self.log_view:
      s += "-log_view\n"
    return s
  
  def interpolate(self, points_coords:np.ndarray) -> np.ndarray:
    ndata = np.int32(self.data_coords.shape[0])
    # C functions expects 1D arrays in C order
    data_coords = self.data_coords.view().astype(np.float64)
    data_coords.shape = (ndata*self.dim)
    data_values = self.data_values.view().astype(np.float64)
    data_values.shape = (ndata*self.nfields)

    npoints = np.int32(points_coords.shape[0])
    points_coords = points_coords.view().astype(np.float64)
    points_coords.shape = (npoints*self.dim)
    points_values = np.zeros(npoints*self.nfields, dtype=np.float64)

    lrb.cfuncs["lagrbf_interpolate"](
      ndata,
      data_coords,
      data_values,
      npoints,
      points_coords,
      points_values,
      self.options_file.encode("utf-8")
    )
    points_values.shape = (npoints, self.nfields)
    return points_values

def test():
  import pyvista as pvs

  sphere = pvs.Sphere(radius=2.2, theta_resolution=50, phi_resolution=50)
  values = np.zeros((sphere.n_points, 2), dtype=np.float64) # dummy values
  
  x = sphere.points[:,0]
  y = sphere.points[:,1]
  z = sphere.points[:,2]
  # create a dummy field that varies with x, y, z
  values[:,0] = x + y + z
  values[:,1] = x*y + z*x + y*z
  sphere["f0"] = values[:,0]
  sphere["f1"] = values[:,1]
  sphere.save("sphere.vtp")

  interpolator = LagRBFInterpolator(sphere.points, values, log_view=True, partitions=np.array([8,8,8], dtype=np.int32))
  sub_sphere = pvs.Sphere(radius=2.2, theta_resolution=20, phi_resolution=20)
  sub_points = sub_sphere.points
  interpolated_values = interpolator.interpolate(sub_points)
  
  output = pvs.PolyData(sub_points)
  output["f0"] = interpolated_values[:,0]
  output["f1"] = interpolated_values[:,1]
  output.save("sub_sphere_interpolated.vtp")

  return


if __name__ == "__main__":
  test()