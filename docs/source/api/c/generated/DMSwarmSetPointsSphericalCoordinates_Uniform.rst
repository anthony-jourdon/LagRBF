DMSwarmSetPointsSphericalCoordinates_Uniform
============================================

.. c:function:: static PetscErrorCode DMSwarmSetPointsSphericalCoordinates_Uniform(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)

   Set point coordinates in a `DMSWARM` on a regular (ijk) spherical grid

   **Collective**

   **Input Parameters**

   - ``sw`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - the `DMSWARM`
   - ``min`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - minimum coordinate values in the r, theta, phi directions (array of length dim)
   - ``max`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - maximum coordinate values in the r, theta, phi directions (array of length dim)
   - ``npoints`` (`PetscInt <https://petsc.org/release/manualpages/Sys/PetscInt/>`_) - number of points in each spatial direction (array of length dim)
   - ``mode`` (``InsertMode``) - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)

   **Notes**

   - When using mode = `INSERT_VALUES`, this method will reset the number of particles in the `DMSWARM`
   - to be `npoints[0]` x `npoints[1]` (2D) or `npoints[0]` x `npoints[1]` x `npoints[2]` (3D). When using mode = `ADD_VALUES`,
   - new points will be appended to any already existing in the `DMSWARM`

   **Source**

   - ``src/swarm/swarm_utils.c``
