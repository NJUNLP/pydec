# PYDEC.SUM
> pydec.sum(input, *, dtype=None) →  {{{pydec_Composition}}}

Returns the sum of all elements of each component in the `input` composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

**Keyword Arguments:**
* **dtype** (*{{{torch_dtype}}}, optional*) - the desired data type of returned composition. If specified, the input composition is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.

Example:
```python
>>> c = pydec.Composition(torch.randn(3, 2))
>>> c
"""
composition{
  components:
    tensor([ 0.0164, -0.7584]),
    tensor([-1.0275, -0.5430]),
    tensor([ 1.8873, -0.1644]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.sum(c)
"""
composition{
  components:
    tensor(-0.7421),
    tensor(-1.5705),
    tensor(1.7229),
  residual:
    tensor(0.)}
"""
```

> pydec.sum(input, dim, keepdim=False, *, dtype=None) →  {{{pydec_Composition}}}

Returns the sum of each row of each component in the `input` composition in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.

If `keepdim` is *True*, the output composition is of the same size as `input` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [torch.squeeze()](https://pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze)), resulting in the output composition having 1 (or `len(dim)`) fewer dimension(s).

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}} or tuple of ints*) – the dimension or dimensions to reduce.
* **keepdim** (*{{{python_bool}}}*) - whether the output composition has `dim` retained or not.

**Keyword Arguments:**
* **dtype** (*{{{torch_dtype}}}, optional*) - the desired data type of returned composition. If specified, the input composition is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.

Example:
```python
>>> c = pydec.Composition(torch.randn(2, 3, 4))
>>> c
"""
composition{
  components:
    tensor([[-1.3777,  0.0411,  0.8618, -0.7705],
            [-1.4941, -2.5065, -0.3250, -0.6044],
            [ 0.3607,  0.7738,  2.1104, -0.2744]]),
    tensor([[ 2.3417,  0.1460,  0.1450, -0.5371],
            [ 0.1098,  0.6213, -0.6196,  1.5271],
            [-0.7728, -1.0843,  0.6843, -3.0325]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> pydec.sum(c, 1)
"""
composition{
  components:
    tensor([-1.2453, -4.9300,  2.9704]),
    tensor([ 2.0956,  1.6386, -4.2053]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.sum(c, (1, 0))
"""
composition{
  components:
    tensor(-3.2050),
    tensor(-0.4710),
  residual:
    tensor(0.)}
"""
```