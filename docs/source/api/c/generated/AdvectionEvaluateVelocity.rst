AdvectionEvaluateVelocity
=========================

.. c:function:: PetscErrorCode AdvectionEvaluateVelocity(LagRBFCtx ctx, DM swi, DM swd)

   Evaluate the velocity field at the current particle positions.

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - The `LagRBFCtx` context
   - ``swi`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - The `DM` swarm containing the interpolation points (particles)
   - ``swd`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - The `DM` swarm containing the data points (velocity field). Can be NULL if using an analytical velocity field.

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function uses the velocity type specified in the `AdvectionCtx` to either:
   - 1) Use RBF interpolation to evaluate the velocity at the particle positions
   - 2) Use a user-defined function to evaluate the velocity at the particle positions

   **Source**

   - ``src/advection/advection.c``
