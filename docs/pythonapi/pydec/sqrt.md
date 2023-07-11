# PYDEC.SQRT
> pydec.sqrt(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.sqrt()](https://pytorch.org/docs/stable/generated/torch.sqrt.html#torch.sqrt).

This is an OVF and the result depends on the currently enabled decomposition algorithm. See {{#auto_link}}pydec.decOVF with_parentheses:false{{/auto_link}} for details.


**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.
* **ref** (*{{{torch_Tensor}}}, optional*) - the reference tensor of `input`. If not given, it is obtained by `input.recovery`.

Example:
```python
>>> c = pydec.Composition(torch.rand(2,3))
>>> c
"""
composition{
  components:
    tensor([0.0623, 0.0642, 0.6957]),
    tensor([0.6990, 0.6193, 0.0525]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.sqrt(c)
"""
composition{
  components:
    tensor([0.0714, 0.0777, 0.8043]),
    tensor([0.8011, 0.7491, 0.0607]),
  residual:
    tensor([0., 0., 0.])}
"""
```