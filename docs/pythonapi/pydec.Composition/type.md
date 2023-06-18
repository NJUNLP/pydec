# PYDEC.COMPOSITION.TYPE
> Composition.type(dtype=None, non_blocking=False) → {{{python_str}}} or {{{pydec_Composition}}}

Returns the type if *dtype* is not provided, else casts this object to the specified type.

If this is already of the correct type, no copy is performed and the original object is returned.

**Parameters:**

* **dtype** (*{{{torch_dtype}}} or {{{python_str}}}*) - The desired type.
* **non_blocking** (*{{{python_bool}}}*) - If *True* and the source is in pinned memory and destination is on the GPU or vice versa, the copy is performed asynchronously with respect to the host. Otherwise, the argument has no effect. Default: *False*.