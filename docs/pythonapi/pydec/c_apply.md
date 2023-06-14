# PYDEC.C_APPLY
> pydec.c_apply(input, callable) →  {{pydec_Composition}}

Applies the function `callable` to each component in the composition.

!> `callable` should accept a tensor and return the processed tensor.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) - the input composition.
* **callable** (*{{{python_callable}}}*) - the function applied to each component.

Example:
```python
>>> c = pydec.Composition(torch.randn(2, 4))
>>> c
"""
composition{
  components:
    tensor([-0.6218,  0.9365,  0.5795, -0.0667]),
    tensor([2.2072, 1.2619, 1.0789, 1.0163]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> pydec.c_apply(c, torch.cos)
"""
composition{
  components:
    tensor([0.8128, 0.5926, 0.8367, 0.9978]),
    tensor([-0.5943,  0.3040,  0.4723,  0.5265]),
  residual:
    tensor([1., 1., 1., 1.])}
"""
>>> pydec.c_apply(c, lambda x: x.square())
"""
composition{
  components:
    tensor([0.3867, 0.8771, 0.3358, 0.0044]),
    tensor([4.8718, 1.5925, 1.1640, 1.0328]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
```
