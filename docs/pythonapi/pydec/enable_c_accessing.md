# PYDEC.ENABLE_C_ACCESSING
> CLASS pydec.enable_c_accessing

Context-manager that enables indexing and slicing of components. If indexing and slicing of components is enabled, the first dimension of indices is used to access components. If indexing a single component, a tensor is returned. If indexing multiple components, they are returned as a composition. Also, `len()` returns the number of components and `iter()` returns an iterator that traverses each component.

Also functions as a decorator.

Example:
```python
>>> c = pydec.Composition(torch.rand((3,4)))
>>> c
"""
composition{
  components:
    tensor([0.8693, 0.8431, 0.8175, 0.5215]),
    tensor([0.0380, 0.0870, 0.5603, 0.5290]),
    tensor([0.3138, 0.4371, 0.4443, 0.5056]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> c[0]
"""
composition{
  components:
    tensor(0.8693),
    tensor(0.0380),
    tensor(0.3138),
  residual:
    tensor(0.)}
"""
>>> len(c)
"""
4
"""
>>> with pydec.enable_c_accessing():
...     c[0]
...     c[0:1]
...     c[0:2, :2]
...     len(c)
"""
tensor([0.8693, 0.8431, 0.8175, 0.5215])
composition{
  components:
    tensor([0.8693, 0.8431, 0.8175, 0.5215]),
  residual:
    tensor([0., 0., 0., 0.])}
composition{
  components:
    tensor([0.8693, 0.8431]),
    tensor([0.0380, 0.0870]),
  residual:
    tensor([0., 0.])}
3
"""
```