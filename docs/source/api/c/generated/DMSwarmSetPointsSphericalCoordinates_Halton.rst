DMSwarmSetPointsSphericalCoordinates_Halton
===========================================

.. c:function:: static PetscErrorCode DMSwarmSetPointsSphericalCoordinates_Halton(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)

   Set point coordinates in a `DMSWARM` on a spherical grid using Halton sequence

   **Collective**

   **Input Parameters**

   - ``sw`` - the `DMSWARM`
   - ``min`` - minimum coordinate values in the r, theta, phi directions (array of length dim)
   - ``max`` - maximum coordinate values in the r, theta, phi directions (array of length dim)
   - ``npoints`` - number of points in each spatial direction (array of length dim)
   - ``mode`` - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)

   **Notes**

   - When using mode = `INSERT_VALUES`, this method will reset the number of particles in the `DMSWARM`
   - to be `npoints[0]` x `npoints[1]` (2D) or `npoints[0]` x `npoints[1]` x `npoints[2]` (3D). When using mode = `ADD_VALUES`,
   - new points will be appended to any already existing in the `DMSWARM`

   **Source**

   - ``/home/jourdon/Software/lagrangian-rbf/src/swarm/swarm_utils.c``
