Interface_Interpolation
=======================

.. c:function:: PetscErrorCode Interface_Interpolation(LagRBFCtx ctx)

   Interpolate values from data swarm points to interpolation swarm points.

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - the LagRBF context

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - This function performs interpolation on all fields registered in the data swarm simultaneously.
   - Calls :c:func:`RBFInterpolation` to perform the interpolation.

   **See also**

   - :c:func:`RBFInterpolation`

   **Source**

   - ``src/interface/interface_interp.c``
