lagrbf_create
=============

.. c:function:: void *lagrbf_create()

   Creates the :c:type:`LagRBFCtx` context and initializes the data structures from options. To be called from Python.

   **Collective**

   **Output Parameters**

   - ``ctx`` (``void *``) - pointer to the LagRBFCtx context to be created

   **Notes**

   - This function initializes the data structures in the memory. User must call :c:func:`lagrbf_destroy` to free the memory.
   - This function is a wrapper for :c:func:`Interface_LagRBFCreate` to be called from Python.

   **See also**

   - :c:func:`lagrbf_destroy`
   - :c:func:`Interface_LagRBFCreate`

   **Source**

   - ``src/interface/interface_ctx.c``
