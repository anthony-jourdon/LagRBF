#!/usr/bin/python3.13
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-64-bit-indices=0',
    '--with-blas-dir=/usr/lib/x86_64-linux-gnu/blas',
    '--with-cc=mpicc',
    '--with-cmake-dir=/usr/lib/x86_64-linux-gnu/cmake',
    '--with-cxx=mpicxx',
    '--with-debugging=0',
    '--with-fc=0',
    '--with-lapack-dir=/usr/lib/x86_64-linux-gnu/lapack',
    '--with-memalign=32',
    '--with-mpiexec=mpiexec',
    '--with-shared-libraries=true',
    '--with-single-library=true',
    '--with-zlib=1',
    'COPTFLAGS=-march=native -g -O3',
    'CXXOPTFLAGS=-march=native -g -O3',
    'FOPTFLAGS=-march=native -g -O3',
    'PETSC_ARCH=arch-linux-openmpi-opt-32bits',
  ]
  configure.petsc_configure(configure_options)
