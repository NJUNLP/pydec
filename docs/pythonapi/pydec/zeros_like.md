# PYDEC.ZEROS_LIKE
> pydec.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)  →  {{{pydec_Composition}}}

Returns a composition filled with the scalar value *0*, with the same size as `input`. `pydec.zeros_like(input)` is equivalent to `pydec.zeros(input.size(), input.numc(), dtype=input.dtype, layout=input.layout, device=input.device)`.

**Parameters:**

* **input** ({{{pydec_Composition}}}) - the size of `input` will determine size of the output composition.

**Keyword Arguments:**

* **dtype** (*{{{torch_dtype}}}, optional*) - the desired data type of returned composition. Default: if **None**, defaults to the dtype of `input`.
* **layout** (*{{{torch_layout}}}, optional*) – the desired layout of returned composition. Default: if **None**, defaults to the layout of `input`.
* **device** (*{{{torch_device}}}, optional*) - the desired device of returned composition. Default: if **None**, defaults to the device of `input`.
* **requires_grad** (*{{{python_bool}}}, optional*) - If autograd should record operations on the returned composition. Default: **False**.
* **memory_format** (*{{{torch_memory_format}}}, optional*) – the desired memory format of returned composition. Default: **torch.preserve_format**.

Example:
```python
>>> input = pydec.zeros((2, 3), c_num=1)
>>> pydec.zeros_like(input)
"""
composition{
  components:
    tensor([[0., 0., 0.],
            [0., 0., 0.]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
```
