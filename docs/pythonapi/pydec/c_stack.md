# PYDEC.C_STACK
> pydec.c_stack(components, *, out=None) →  {{{pydec_Composition}}}

Concatenates a sequence of tensors along a new dimension and returns their composition.

All tensors need to be of the same size.

**Parameters:**

* **components** (*sequence of Tensor*) – sequence of tensors to concatenate.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> t = torch.rand(2, 4)
>>> t
"""
tensor([[0.0121, 0.4041, 0.5754, 0.8529],
        [0.0423, 0.3423, 0.7190, 0.9070]])
"""
>>> pydec.c_stack((t, t, t))
"""
composition{
  components:
    tensor([[0.0121, 0.4041, 0.5754, 0.8529],
            [0.0423, 0.3423, 0.7190, 0.9070]]),
    tensor([[0.0121, 0.4041, 0.5754, 0.8529],
            [0.0423, 0.3423, 0.7190, 0.9070]]),
    tensor([[0.0121, 0.4041, 0.5754, 0.8529],
            [0.0423, 0.3423, 0.7190, 0.9070]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
```
