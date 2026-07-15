RBFInterpolation
================

.. c:function:: PetscErrorCode RBFInterpolation(LagRBFCtx ctx, DM swi, DM swd, const char *fieldname)

   Interpolate values of one field from data swarm points to interpolation swarm points

   **Collective**

   **Input Parameters**

   - ``ctx`` (:c:type:`LagRBFCtx`) - the LagRBF context
   - ``swi`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - interpolation swarm (target points)
   - ``swd`` (`DM <https://petsc.org/release/manualpages/DM/DM/>`_) - data swarm (source points)
   - ``fieldname`` (``const char *``) - name of the field to interpolate

   **Output Parameters**

   - ``PETSC_SUCCESS`` - if successful

   **Notes**

   - Backend is selected at compile time:
   - ``__AVX512F__``: AVX-512 implementation
   - ``__AVX__``: AVX-256 implementation
   - ``fallback``: Scalar implementation

   **See also**

   - `DM <https://petsc.org/release/manualpages/DM/DM/>`_
   - `DMSWARM <https://petsc.org/release/manualpages/DMSWARM/DMSWARM/>`_
   - `DMSwarmGetField <https://petsc.org/release/manualpages/DMSWARM/DMSwarmGetField/>`_
   - `DMSwarmRestoreField <https://petsc.org/release/manualpages/DMSWARM/DMSwarmRestoreField/>`_
   - `PetscErrorCode <https://petsc.org/release/manualpages/Sys/PetscErrorCode/>`_

   **Source**

   - ``/home/jourdon/Software/lagrangian-rbf/src/interpolation/interpolation.c``
