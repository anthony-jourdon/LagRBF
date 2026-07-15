AdvectionEvaluateVelocity
=========================

.. c:function:: PetscErrorCode AdvectionEvaluateVelocity(LagRBFCtx ctx, DM swi, DM swd)

   Evaluate the velocity field at the current particle positions.

   **Input Parameters**

   - ``ctx`` - The `LagRBFCtx` context
   - ``swi`` - The `DM` swarm containing the interpolation points (particles)
   - ``swd`` - The `DM` swarm containing the data points (velocity field). Can be NULL if using an analytical velocity field.

   **Notes**

   - This function uses the velocity type specified in the `AdvectionCtx` to either:
   - 1) Use RBF interpolation to evaluate the velocity at the particle positions
   - 2) Use a user-defined function to evaluate the velocity at the particle positions

   **Source**

   - ``/home/jourdon/Software/lagrangian-rbf/src/advection/advection.c``
