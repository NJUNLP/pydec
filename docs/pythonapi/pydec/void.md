# PYDEC.VOID
> pydec.void(*, dtype=None, device=None, requires_grad=False) →  {{{pydec_Composition}}}

Returns an void composition.

!> No underlying data will be created for the composition.

**Keyword Arguments:**

* **dtype** (*{{{torch_dtype}}}, optional*) - the desired data type of returned composition. Default: if **None**, uses a global default (see {{{torch_set_default_tensor_type}}}).
* **device** (*{{{torch_device}}}, optional*) - the desired device of returned composition. Default: if **None**, uses the current device for the default tensor type (see {{{torch_set_default_tensor_type}}}). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** (*{{{python_bool}}}, optional*) - If autograd should record operations on the returned composition. Default: **False**.

Example:
```python
>>> pydec.void()
"""
composition{}
"""
```
