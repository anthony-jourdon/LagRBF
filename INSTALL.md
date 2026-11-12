# Installation Guide for LagRBF

LagRBF is a Radial Basis Function interpolation and Lagrangian point advection library written in C with a Python API interface. 
This guide covers installation for different user types.

## Quick Start

For experienced users on Linux with MPI and linear algebra libraries already installed:

```bash
# 1. Install PETSc (see PETSc Installation section)
# 2. Install LagRBF into a python venv (created if does not exist)
./install.sh -v path/to/python/virtual_env -p path/to/petsc/directory -a name_of_petsc_arch -j 4
```

To add optional dependencies add `-e all`. See help with `./install.sh -h`.

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (development platform), Windows WSL2 (tested), macOS (not tested) 
- **Python**: 3.12+ (earlier versions not tested)
- **Compiler**: GCC 9+ (other compilers not tested)
- **MPI**: A functional MPI implementation (OpenMPI, MPICH, or vendor-specific)
- **Linear Algebra**: BLAS and LAPACK libraries (system or precompiled)
- **Git**: For cloning the repository

### Python Dependencies

Install these Python packages:

- **NumPy** ≥ 2.5 — numerical computing
- **petsc4py** ≥ 3.25.3 — PETSc Python bindings (requires PETSc C library)
- **PyVista** ≥ 0.37 — 3D visualization and VTK interface
- **Sphericart** (optional) — for analytical flow functions

---

## 1. Install PETSc C Library

LagRBF requires [PETSc](https://petsc.org/release/) with specific configuration options. This is the most complex step.

### 1.1. Download PETSc

```bash
# Download PETSc
wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.25.3.tar.gz
# untar the petsc archive
tar xzvf petsc-3.25.3.tar.gz
# move into PETSc directory
cd petsc-3.25.3
```

### 1.2. Configure PETSc

[PETSc](https://petsc.org/release/) configuration is highly machine-dependent. The repository provides example configurations:

**For Ubuntu with OpenMPI and BLAS/LAPACK installed from system's package manager:**

```bash
cd petsc-3.25.3
python3 ../petsc_config/ubuntu.arch-linux-openmpi-opt-32bits.py
```

**For custom configurations:**

See {doc}`petsc_config` for platform-specific guidance. The key required options are:

```bash
--with-memalign=32
--with-shared-libraries=true
--with-single-library=true
```

### 1.3. Build and Install PETSc

```bash
# Build PETSc (this takes 10-30 minutes)
# Normally, when successfully built, PETSc provides you the following instructions:
make PETSC_DIR=path/to/petsc PETSC_ARCH=petsc_arch_name all

# Test installation (optional but recommended)
make PETSC_DIR=path/to/petsc PETSC_ARCH=petsc_arch_name test

# (Optional) Export for later use, if you use the install.sh script this step can be done when running it
export PETSC_DIR=path/to/petsc/directory
export PETSC_ARCH=name_of_petsc_arch  # your chosen PETSC_ARCH name
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

## 2a. LagRBF automated installation (Recommended)

The repository provides an `install.sh` script to automate LagRBF compilation and python package installation:

```bash
./install.sh -v python_venv_name -p /path/to/petsc/directory -a petsc_arch_name -e all -j 4
```

Note that at this step you can use `-p` and `-a` to set `PETSC_DIR` and `PETSC_ARCH` environment variable automatically.

For a lighter installation you can remove `-e` option, this will prevent documentation-required packages to be installed as well as the [Sphericart](https://sphericart.readthedocs.io/en/latest/) package used for analytical benchmarks.

See `./install.sh -h` for available options.

## 2b. Manual step-by-step installation

In case of problems with automated installation, the following step-by-step procedure will install LagRBF C libraries and python package.

> [!WARNING]
> `PETSC_DIR` and `PETSC_ARCH` environment variables need to be set before the following steps.

### 2.1. Create a Python virtual environment

A virtual environment isolates your project dependencies from system-wide packages:

```bash
# Create virtual environment
python3.13 -m venv ~/.venv/lagrbf
source ~/.venv/lagrbf/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

### 2.2. Install petsc4py

Once [PETSc](https://petsc.org/release/) is built and tested, build the Python bindings:

```bash
python -m pip install $PETSC_DIR/src/binding/petsc4py

# Verify
python -c "import petsc4py; petsc4py.init()"
```

### 2.3. Clone and Build LagRBF C Library

```bash
# Clone LagRBF repository
git clone https://github.com/anthony-jourdon/LagRBF.git
cd LagRBF

# Compile C library and tests (builds into $PETSC_ARCH/ directory)
make all -j4

# Verify build
ls $PETSC_ARCH/lib/liblagrbf.*
```

(Optional) Adjust compilation flags in `makefile.arch`:

```bash
# Edit to tune optimization level etc.
nano makefile.arch
```

Normally, AVX-related flags are automatically added if found.

### 2.4. Install LagRBF Python Package

The Python package can be installed in two ways:

**Option A: Development Mode (Recommended for Users/Developers)**

```bash
pip install -e "python_api[all]"
```

This creates a symlink to the Python package, so changes to Python code are immediately reflected. C library updates require recompiling with `make all`.

**Option B: Standard Installation (For Production)**

```bash
pip install python_api
```
or 
```bash
pip install "python_api[analytical]"
```
to add [Sphericart](https://sphericart.readthedocs.io/en/latest/) package and enable benchmark functionalities.

### 2.5. Configure Environment for Future Sessions

Create an environment setup file to avoid re-entering paths each time:

```bash
cat > path/to/venv/bin/lagrbf_env.sh << 'EOF'
# LagRBF Environment Setup
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=petsc_arch_name
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
export LAGRBF_DIR=/path/to/LagRBF
EOF

# Add it to the venv activation script
cat >> path/to/venv/bin/activate << 'EOF'
if [ -f "${BASH_SOURCE%/*}/lagrbf_env.sh" ]; then
    source "${BASH_SOURCE%/*}/lagrbf_env.sh" 2>/dev/null
elif [ -f "$(dirname "$0")/lagrbf_env.sh" ]; then
    source "$(dirname "$0")/lagrbf_env.sh" 2>/dev/null
fi
EOF
```

## Troubleshooting

### PETSc Configuration Fails

**Issue**: `--with-blas-dir` or `--with-lapack-dir` path not found

**Solution**: Locate your BLAS/LAPACK installation:

```bash
# Ubuntu/Debian
dpkg -L libblas-dev | grep lib
apt-cache search blas

# macOS
brew list openblas
```

Update the configuration script with correct paths.

### petsc4py Import Error

**Issue**: `ImportError: libpetsc.so.3.25: cannot open shared object file`

**Solution**: Ensure `LD_LIBRARY_PATH` includes PETSc libraries:

```bash
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
python -c "import petsc4py"
```

### `make all` Fails

**Issue**: `PETSC_DIR` or `PETSC_ARCH` not set

**Solution**:

```bash
export PETSC_DIR=/full/path/to/petsc
export PETSC_ARCH=arch-linux-openmpi-opt-32bits
echo "PETSC_DIR=$PETSC_DIR, PETSC_ARCH=$PETSC_ARCH"  # Verify
make clean
make all -j4
```

### MPI or Compiler Not Found

**Issue**: `mpicc: command not found`

**Solution**: Load MPI module on HPC systems:

```bash
module avail mpi
module load openmpi/4.1.2  # or your cluster's module name
```

## Validation

After installation, verify the setup:

```bash
python -c "
import lagrbfpy
print('✓ lagrbfpy imported successfully')

from lagrbfpy import LagRBFCtx, PointSwarm
print('✓ Core classes available')

from lagrbfpy import HarmonicFlow
print('✓ Optional sphericart dependency available')
" 2>&1 || echo "⚠ Some features unavailable; install sphericart for full functionality"
```

## Uninstallation

To completely remove LagRBF:

```bash
# Uninstall Python package
pip uninstall lagrbfpy

# Remove virtual environment
rm -rf path/to/virtual/env

# Remove PETSc (optional)
rm -rf /path/to/petsc

# Remove LagRBF source
rm -rf /path/to/lagrangian-rbf
```

---

## Next Steps

- **Python API Documentation**: See [documentation](#) (TODO)
- **Examples**: Browse [`python_api/scripts/`](python_api/scripts/)
- **Contributing**: See [`DEVELOPMENT.md`](DEVELOPMENT.md)
- **Issues**: Report on [GitHub Issues](https://github.com/anthony-jourdon/LagRBF/issues)

---

## Getting Help

1. Check this guide and troubleshooting section
2. Review [PETSc Installation Guide](https://petsc.org/release/install/)
3. Check {doc}`petsc_config` for platform-specific notes
4. Open a [GitHub Issue](https://github.com/anthony-jourdon/lagrangian-rbf/issues)

**When reporting issues, include**:
- OS and compiler version (`uname -a`, `gcc --version`)
- Python version (`python --version`)
- PETSc version and architecture
- Full error message and last 20 lines of build output
