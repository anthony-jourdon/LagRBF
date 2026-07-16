AdvectionComputeTimestep_CFL
============================

.. c:function:: PetscErrorCode AdvectionComputeTimestep_CFL(LagRBFCtx ctx, DM swarm, PetscReal *_dt)

   Compute the timestep based on the CFL condition.

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - The `LagRBFCtx` context
   - ``swarm`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - The `DM` swarm containing the interpolation points (particles)

   **Output Parameters**

   - ``_dt`` (`PetscReal <https://petsc.org/release/manualpages/Sys/PetscReal/>`_) - Pointer to store the computed timestep

   **Notes**

   - This function computes the timestep based on the CFL condition, which is determined by the maximum velocity and the support radius of the partition of unity.
   - The computed timestep is returned through the `_dt` pointer.

   **Source**

   - ``src/advection/advection.c``
