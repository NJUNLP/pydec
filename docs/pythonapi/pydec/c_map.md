# PYDEC.C_MAP
> pydec.c_map(input, other, callable) →  {{pydec_Composition}}

Applies `callable` for each component in `input` and the given composition.

!> `callable` should accept two tensor and return a processed tensor.

The `callable` should have the signature:
```python
def callable(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) - the input composition.
* **callable** (*{{{python_callable}}}*) - the function applied to each pair of components in `input` and `other`.

Example:
```python
>>> a = pydec.Composition(torch.randn(2, 3))
>>> a
"""
composition{
  components:
    tensor([ 1.5280, -1.3948, -0.0360]),
    tensor([-1.2433, -0.0679, -0.2001]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> b = pydec.Composition(torch.randn(2, 3))
>>> b
"""
composition{
  components:
    tensor([ 1.0418, -1.0125, -0.1757]),
    tensor([2.1716, 0.2520, 0.7730]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.c_map(a, b, torch.cross)
"""
composition{
  components:
    tensor([ 0.2086,  0.2310, -0.0940]),
    tensor([-0.0020,  0.5266, -0.1659]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.c_map(a, b, lambda x, y: x + y.sum())
"""
composition{
  components:
    tensor([4.5781, 1.6553, 3.0141]),
    tensor([1.8068, 2.9822, 2.8500]),
  residual:
    tensor([0., 0., 0.])}
"""
```
