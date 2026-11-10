#!/usr/bin/python3.12
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-64-bit-indices=0',
    '--with-blaslapack-dir=/home/jourdon/software/mkl/intel/oneapi/mkl/2025.1/',
    '--with-cc=/usr/lib64/mpich/bin/mpicc',
    '--with-cxx=/usr/lib64/mpich/bin/mpicxx',
    '--with-debugging=0',
    '--with-fc=0',
    '--with-memalign=32',
    '--with-mpi-include=/usr/include/mpich-x86_64',
    '--with-mpi-lib=/usr/lib64/mpich/lib/libfmpich.so /usr/lib64/mpich/lib/libmpich.so /usr/lib64/mpich/lib/libmpichcxx.so',
    '--with-mpiexec=/usr/lib64/mpich/bin/mpiexec',
    '--with-shared-libraries=true',
    '--with-single-library=true',
    '--with-zlib=1',
    'COPTFLAGS=-march=native -O3',
    'CXXOPTFLAGS=-march=native -O3',
    'FOPTFLAGS=-march=native -O3',
    'PETSC_ARCH=arch-linux-c-mpich-opt-32bits',
  ]
  configure.petsc_configure(configure_options)
