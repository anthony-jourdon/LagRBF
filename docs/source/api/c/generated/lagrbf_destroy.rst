lagrbf_destroy
==============

.. c:function:: void lagrbf_destroy(void *ctx)

   Destroys the :c:type:`LagRBFCtx` context and frees the memory.

   **Collective**

   **Notes**

   - This function frees the memory allocated for the LagRBFCtx context. User must call this function when they are done with the context.
   - This function is a wrapper for :c:func:`LagRBFCtxDestroy` to be called from Python.

   **See also**

   - :c:func:`lagrbf_create`
   - :c:func:`LagRBFCtxDestroy`
   - :c:func:`Interface_LagRBFCreate`

   **Source**

   - ``src/interface/interface_ctx.c``
