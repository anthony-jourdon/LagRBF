# PETSc Configuration Guide

This directory contains example [PETSc](https://petsc.org/release/) configuration scripts for different platforms and MPI implementations. These scripts automate the [PETSc](https://petsc.org/release/) installation process with the required options for LagRBF.

## Overview

[PETSc](https://petsc.org/release/) (Portable, Extensible Toolkit for Scientific computing) is a large-scale parallel linear and nonlinear equation solver library. LagRBF requires specific PETSc configuration options:

```bash
--with-memalign=32              # Memory alignment for vectorization
--with-shared-libraries=true    # Build shared libraries
--with-single-library=true      # Build single monolithic library
```

In addition to these, you may need to customize:
- MPI compiler and launcher
- BLAS/LAPACK locations
- Optimization flags
- Debug symbols
- Architecture-specific flags

## Available Configurations

### 1. Ubuntu with OpenMPI

**File**: `ubuntu.arch-linux-openmpi-opt-32bits.py`

**Suitable for**:
- OpenMPI (installed via `apt install openmpi-bin libopenmpi-dev`)
- System BLAS/LAPACK libraries
- x86-64 architecture

**Quick Start**:

```bash
# Install prerequisites
sudo apt-get install -y openmpi-bin libopenmpi-dev libblas-dev liblapack-dev libcmake-dev

# Configure PETSc
cd path/to/petsc/directory
python3 /path/to/LagRBF/petsc_config/ubuntu.arch-linux-openmpi-opt-32bits.py

# Build following PETSc prompted instructions
```

**Key Settings**:
- Compiler: `mpicc` (OpenMPI wrapper around GCC)
- BLAS/LAPACK: System directories `/usr/lib/x86_64-linux-gnu/`
- Optimization: `-march=native -O3`
- Architecture name: `arch-linux-openmpi-opt-32bits`

### 2. 128 core SUSE machine with MPICH

**File**: `pilbara.arch-linux-c-mpich-opt-32bits.py`

**Suitable for**:
- HPC clusters with MPICH
- Environment-specific BLAS/LAPACK

**Quick Start**:

```bash
# Load MPICH module (if on HPC)
module load mpich

# Configure PETSc
cd path/to/petsc/directory
python3 /path/to/LagRBF/petsc_config/pilbara.arch-linux-c-mpich-opt-32bits.py

# Build following PETSc prompted instructions
```

**Key Settings**:
MPI libraries and include location directly passed using:
```
--with-mpi-include=/usr/include/mpich-x86_64
--with-mpi-lib=/usr/lib64/mpich/lib/libfmpich.so /usr/lib64/mpich/lib/libmpich.so /usr/lib64/mpich/lib/libmpichcxx.so
```

## Customizing for Your System

If none of the provided configurations match your system, you can create a custom configuration. Here's a template:

### Creating a Custom Configuration

1. **Copy and rename** an existing configuration:

```bash
cp ubuntu.arch-linux-openmpi-opt-32bits.py custom-system.py
nano custom-system.py
```

2. **Modify key parameters**:

```python
#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    # Required options for LagRBF
    '--with-memalign=32',
    '--with-shared-libraries=true',
    '--with-single-library=true',
    
    # Your system-specific settings
    '--with-cc=mpicc',           # Your MPI C compiler
    '--with-cxx=mpicxx',         # Your MPI C++ compiler
    '--with-fc=0',               # Disable Fortran if not needed
    '--with-debugging=0',        # 0 for optimized, 1 for debug
    
    # BLAS and LAPACK
    '--with-blas-dir=/path/to/blas',
    '--with-lapack-dir=/path/to/lapack',
    
    # MPI launcher
    '--with-mpiexec=mpiexec',
    
    # Optional: Custom flags
    'COPTFLAGS=-march=native -g -O3',
    'CXXOPTFLAGS=-march=native -g -O3',
    
    # Architecture identifier (shows in $PETSC_ARCH after build)
    'PETSC_ARCH=arch-custom-opt-32bits',
  ]
  configure.petsc_configure(configure_options)
```

3. **Run your custom configuration**:

```bash
cd path/to/petsc/directory
python3 path/to/LagRBF/petsc_config/custom-system.py
# Build 
```


## Troubleshooting Common Issues

### Issue: BLAS/LAPACK libraries not found

**Error**: `Could not find BLAS installation`

**Solution**:

Find your BLAS/LAPACK installation:

```bash
# Ubuntu
dpkg -L libblas-dev | grep lib
# Output: /usr/lib/x86_64-linux-gnu/libblas.so.3
# Then use: --with-blas-dir=/usr/lib/x86_64-linux-gnu

# macOS (using Homebrew)
brew install openblas lapack
brew --prefix openblas  # e.g., /usr/local/opt/openblas

# HPC clusters
module avail blas
module load blas/intel-mkl  # or your cluster's BLAS
```

### Issue: MPI compiler not in PATH

**Error**: `Could not find C compiler mpicc`

**Solution**:

Load the MPI module (on HPC systems):

```bash
module avail mpi
module load openmpi/4.1.2  # or MPICH, Intel MPI, etc.
which mpicc  # Verify it's now available
```

Or install MPI locally:

```bash
# Ubuntu/Debian - OpenMPI
sudo apt-get install openmpi-bin libopenmpi-dev

# macOS - Homebrew
brew install open-mpi
```

### Issue: Python 3 not found

**Error**: `python3 not found`

**Solution**:

Ensure Python 3 is installed and the shebang in the config script is correct:

```bash
# Check Python installation
python3 --version

# Update config script shebang if needed
sed -i 's/#!/usr\/bin\/python3.13/#!/usr\/bin\/python3/' custom-system.py
```

## Advanced Configuration Options

### Debug vs. Optimized Builds

**For development/debugging**:

```python
'--with-debugging=1',
'COPTFLAGS=-g -O0',  # No optimization, include debug symbols
'PETSC_ARCH=arch-linux-openmpi-debug-32bits',
```

**For production**:

```python
'--with-debugging=0',
'COPTFLAGS=-march=native -g -O3',  # Optimized with symbols
'PETSC_ARCH=arch-linux-openmpi-opt-32bits',
```

### Enable MKL (Intel Math Kernel Library)

```python
'--with-blas-lapack-dir=$MKLROOT',
'--with-mkl_pardiso=1',  # Optional: for sparse direct solver if available
```

### Using Specific Linear Algebra Libraries

**AMD BLIS/LIBFLAME**:

```python
'--with-blas-dir=/opt/blis',
'--with-lapack-dir=/opt/libflame',
```

## Building petsc4py

Once PETSc is successfully built and verified:

```bash
cd $PETSC_DIR
python3 -m pip install src/binding/petsc4py/
```

Verify petsc4py:

```bash
python3 -c "
import petsc4py
petsc4py.init()
from petsc4py import PETSc
print(f'✓ petsc4py version: {petsc4py.__version__}')
print(f'✓ PETSc version: {PETSc.__version__}')
"
```


## Contributing Custom Configurations

If you've successfully configured PETSc for a specific platform or cluster, consider contributing your configuration:

1. Create a configuration file: `<site>-<mpi>-<compiler>-<profile>-<bits>.py`
2. Test it thoroughly
3. Submit a pull request with:
   - Configuration script
   - Documentation of the platform/cluster
   - Any special considerations or prerequisites

This helps other users with similar systems!

---

## References

- [PETSc Installation Guide](https://petsc.org/release/install/)
- [PETSc Configuration Options](https://petsc.org/release/install/install/)
- {doc}`overview` - LagRBF overview and quick start
- {doc}`installation` - Main installation guide
