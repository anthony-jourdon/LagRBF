# LagRBF

LagRBF is a software developed to interpolate and advect Lagrangian points in 2- and 3-dimensions using radial basis function (RBF) with the partition of unity (PU) method.

The advection is performed using an explicit integration scheme that can be either:

- Euler forward
- Runge-Kutta 2
- Runge-Kutta 4 (default)
- Dormand-Prince 4(5)

The advection velocity can either be interpolated from a dataset or provided from an analytical function describing the velocity vector at all points in space.

The software is written in C and interfaced with Python, which is the recommended approach to use it.

## Installation
### Python
A functional Python is required. 
The software has been developed with python3.13, the compatibility with earlier Python's versions is not guaranteed. 
The following Python's module are required:

- [NumPy](https://numpy.org/)
- [PyVista](https://pyvista.org/)
- [petsc4py](https://petsc.org/release/petsc4py/index.html) (Requires [PETSc](https://petsc.org/release/), see below)
- [Sphericart](https://sphericart.readthedocs.io/en/latest/) (optional, for Cartesian Spherical Harmonics)

### Installing PETSc
LagRBF heavily relies on [PETSc](https://petsc.org/release/). 
No specific external packages are required for [PETSc](https://petsc.org/release/) except a functional MPI for parallel computing.
In addition to the required [PETSc](https://petsc.org/release/) configure options, the following are necessary:
```
--with-memalign=32
--with-shared-libraries=true
--with-single-library=true
```

[PETSc](https://petsc.org/release/) configuration files examples can be found in **TODO: ADD PETSC CONFIG FILE**.

Once [PETSc](https://petsc.org/release/) is installed, your system will need to know where [PETSc](https://petsc.org/release/) has been installed, set:
``` shell
export PETSC_DIR=path/to/petsc/dir
export PETSC_ARCH=name_of_petsc_arch
```
then, build `petsc4py` from the [PETSc](https://petsc.org/release/) directory:
``` shell
python -m pip install src/binding/petsc4py
```

### Recommendations
It is recommended to work with a Python [virtual environment](https://docs.python.org/3/library/venv.html) to avoid any conflict with system-wide Python's modules and dependencies.
A virtual environment can be created with:
``` shell
python3.XX -m venv /path/to/new/virtual/environment
```

In addition, to simplify path management, one can create a file to register environment variables that can be loaded before using the software such that:
``` shell
export PETSC_DIR=path/to/petsc/dir
export PETSC_ARCH=name_of_petsc_arch
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PETSC_DIR/lib/petsc/bin:/path/to/LagRBF/python_utils:$PYTHONPATH
```    

## Compilation
Once [PETSc](https://petsc.org/release/) is installed, compile with:

> make all

the option `-jN` can be used to compile using multithreading.
A directory with the same name than `$PETSC_ARCH` should be created and contains the compiled objects, libraries and executables.

The `Makefile` should not be edited, but the `makefile.arch` can. 
Compilation flags can be added here.

## Documentation
The documentation of the Python API can be found here **TODO**.
Note that the documentation of the C API is still in progress.