# PYDEC.UNSQUEEZE
> pydec.unsqueeze(input, dim) →  {{{pydec_Composition}}}

Returns a new composition with a dimension of size one inserted at the specified position.

The returned composition shares the same underlying data with this tensor.

A `dim` value within the range `[-input.dim() - 1, input.dim() + 1)` can be used. Negative `dim` will correspond to `unsqueeze()` applied at `dim = dim + input.dim() + 1`.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}}*) - the index at which to insert the singleton dimension.

Example:
```python
>>> c = pydec.zeros(2, 3, c_num=2)
>>> c.size()
"""
torch.Size([2, 3])
"""
>>> pydec.unsqueeze(c, 0).size()
"""
torch.Size([1, 2, 3])
"""
>>> pydec.unsqueeze(c, 1).size()
"""
torch.Size([2, 1, 3])
"""
```