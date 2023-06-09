# PYDEC.C_SUM
> pydec.c_sum(input, *, dtype=None) →  {{{torch_Tensor}}}

Returns the sum of all components in the `input` composition.

?> This will get the recovery of the `input` composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

**Keyword Arguments:**

* **dtype** (*{{{torch_dtype}}}, optional*) - the desired data type of returned tensor. If specified, the input composition is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.


Example:
```python
>>> x = pydec.Composition(torch.randn(2,3))
>>> x
"""
composition{
  components:
    tensor([-1.6062,  0.3914,  0.2406]),
    tensor([0.1674, 0.9185, 0.8581]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.c_sum(x)
"""
tensor([-1.4388,  1.3100,  1.0987])
"""
```
