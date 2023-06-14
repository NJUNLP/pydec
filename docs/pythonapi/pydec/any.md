# PYDEC.ANY
> pydec.any(input) →  {{{torch_Tensor}}}

Tests if any element in `input.recovery` evaluates to *True*. 

?> Equivalent to `torch.any(input.recovery)`

?> This function matches the behaviour of NumPy in returning output of dtype *bool* for all supported dtypes except *uint8*. For *uint8* the dtype of output is *uint8* itself.

!> The composition for the bool tensors is uncommon, make sure you know what you are doing.

Example:
```python
>>> c = pydec.Composition(torch.tensor([[0, 1], [1, 0]], dtype=torch.bool))
>>> c
"""
composition{
  components:
    tensor([False,  True]),
    tensor([ True, False]),
  residual:
    tensor([False, False])}
"""
>>> pydec.any(c)
"""
tensor(True)
"""

>>> c = pydec.Composition(torch.tensor([[0, 1], [0, 0]], dtype=torch.bool))
>>> c
"""
composition{
  components:
    tensor([False,  True]),
    tensor([False, False]),
  residual:
    tensor([False, False])}
"""
>>> pydec.any(c)
"""
tensor(True)
"""
```

> pydec.any(input, dim, keepdim=False, *, out=None) →  {{{torch_Tensor}}}

For each row of `input.recovery` in the given dimension `dim`, returns *True* if any elements in the row evaluate to *True* and *False* otherwise.

If `keepdim` is *True*, the output tensor is of the same size as `input.recovery` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [torch.squeeze()](https://pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze)), resulting in the output tensor having 1 fewer dimension than `input.recovery`.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}}*) – the dimension to reduce.
* **keepdim** (*{{{python_bool}}}*) - whether the output tensor has `dim` retained or not.

**Keyword Arguments:**
* **out** (*{{{torch_Tensor}}}, optional*) - the output tensor.