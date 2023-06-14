# PYDEC.MEAN
> pydec.mean(input, *, dtype=None) →  {{{pydec_Composition}}}

Returns the mean value of all elements of each component in the `input` composition.

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
    tensor([1.0008, 1.3038]),
    tensor([0.0999, 0.2713]),
    tensor([-0.8582,  1.1065]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.mean(c)
"""
composition{
  components:
    tensor(1.1523),
    tensor(0.1856),
    tensor(0.1242),
  residual:
    tensor(0.)}
"""
```

> pydec.mean(input, dim, keepdim=False, *, dtype=None, out=None) →  {{{pydec_Composition}}}

Returns the mean value of each row of each component in the `input` composition in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.

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
    tensor([[-0.4625,  1.5479, -0.5996, -1.6460],
            [-0.2495,  1.0912,  0.3206,  0.1110],
            [-0.9828, -0.9904,  0.7163, -0.4185]]),
    tensor([[ 0.2961,  0.3689,  0.5258,  0.3851],
            [ 0.1365, -0.7479,  1.1476,  0.3190],
            [ 1.6006, -0.5255,  0.8142,  0.7070]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> pydec.mean(c, 1)
"""
composition{
  components:
    tensor([-0.2901,  0.3183, -0.4188]),
    tensor([0.3940, 0.2138, 0.6491]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.mean(c, 1, True)
"""
composition{
  components:
    tensor([[-0.2901],
            [ 0.3183],
            [-0.4188]]),
    tensor([[0.3940],
            [0.2138],
            [0.6491]]),
  residual:
    tensor([[0.],
            [0.],
            [0.]])}
"""
```