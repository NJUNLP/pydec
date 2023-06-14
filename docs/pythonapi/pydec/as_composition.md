# PYDEC.AS_COMPOSITION
> pydec.as_composition(components, residual=None)  →  {{{pydec_Composition}}}

Converts tensor into a composition, sharing data and preserving autograd history if possible. The `components` tensor should contain at least one dimension, where the first dimension is the component dimension and the remaining dimensions are the shape of the composition. If there is a `residual`, its `shape`, `dtype` and `device` must be compatible with the `components`.

!> For efficiency reasons, PyDec never checks if the `components` and `residuals` are compatible. Unless you guarantee their compatibility, errors may occur during subsequent use.

?> To construct a composition with no autograd history by copying tensor, use {{#auto_link}}pydec.Composition{{/auto_link}}.

**Parameters:**

* **components** ({{{torch_Tensor}}}) - Initial component tensor for the composition.

**Keyword Arguments:**

* **residual** (*{{{torch_Tensor}}}, optional*) - Initial residual tensor for the composition. If not given, initialize with *0*.

Example:
```python
>>> components = torch.randn(2, 3)
>>> pydec.as_composition(components)
"""
composition{
  components:
    tensor([0.1527, 0.2952, 0.3012]),
    tensor([-0.6841,  0.1097, -0.1200]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> residual = torch.randn(3)
>>> pydec.as_composition(components, residual)
"""
composition{
  components:
    tensor([0.1527, 0.2952, 0.3012]),
    tensor([-0.6841,  0.1097, -0.1200]),
  residual:
    tensor([-1.3614,  1.0464,  2.7408])}
"""
```
