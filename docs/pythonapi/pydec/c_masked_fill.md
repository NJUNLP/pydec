# PYDEC.C_MASKED_FILL
> pydec.c_masked_fill(input, mask, value) →  {{{pydec_Composition}}}

Fills components of the `input` composition with `value` where `mask` is *True*. The component dimension has the same size as the length of `index`.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **mask** (*BoolTensor*) – the 1-D tensor containing the binary mask to index with.
* **value** (*{{{python_float}}}*) – the value to fill in with.

Example:
```python
>>> c = pydec.zeros((2,), 3)
"""
composition{
  components:
    tensor([0., 0.]),
    tensor([0., 0.]),
    tensor([0., 0.]),
  residual:
    tensor([0., 0.])}
"""
>>> mask = torch.tensor([1, 0, 1])
>>> pydec.c_masked_fill(c, mask, 3.0)
"""
composition{
  components:
    tensor([3., 3.]),
    tensor([0., 0.]),
    tensor([3., 3.]),
  residual:
    tensor([0., 0.])}
"""
```