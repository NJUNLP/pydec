# PYDEC.ZEROS
> pydec.(*size, c_num, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) →  {{{pydec_Composition}}}

Returns a composition whose components filled with the scalar value *0*, with the shape and the component number defined by the variable argument `size` and `c_num`, respectively.

**Parameters:**

* **size** (*{{{python_int}}}...*) - a sequence of integers defining the shape of the output composition. Can be a variable number of arguments or a collection like a list or tuple. You must use the keyword argument to specify `c_num` if `size` is specified by a variable number of arguments.
* **c_num** (*{{{python_int}}}*) - the number of components of the output composition.

**Keyword Arguments:**

* **out** ({{{pydec_Composition}}}, optional) – the output composition.
* **dtype** (*{{{torch_dtype}}}, optional*) – the desired data type of returned composition. Default: if **None**, uses a global default (see {{{torch_set_default_tensor_type}}}).
* **layout** (*{{{torch_layout}}}, optional*) – the desired layout of returned composition. Default: **torch.strided**.
* **device** (*{{{torch_device}}}, optional*) – the desired device of returned composition. Default: if **None**, uses the current device for the default tensor type (see {{{torch_set_default_tensor_type}}}). **device** will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** (*{{{python_bool}}}, optional*) – If autograd should record operations on the returned composition. Default: **False**.

Example:
```python
>>> pydec.zeros((1, 2), 3)
"""
composition{
  components:
    tensor([[0., 0.]]),
    tensor([[0., 0.]]),
    tensor([[0., 0.]]),
  residual:
    tensor([[0., 0.]])}
"""

>>> pydec.zeros(1, 3, c_num=2)
"""
composition{
  components:
    tensor([[0., 0., 0.]]),
    tensor([[0., 0., 0.]]),
  residual:
    tensor([[0., 0., 0.]])}
"""
```