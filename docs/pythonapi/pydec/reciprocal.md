# PYDEC.RECIPROCAL
> pydec.reciprocal(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.reciprocal()](https://pytorch.org/docs/stable/generated/torch.reciprocal.html#torch.reciprocal).

This is an OVF and the result depends on the currently enabled decomposition algorithm. See {{#auto_link}}pydec.decOVF with_parentheses:false{{/auto_link}} for details.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.
* **ref** (*{{{torch_Tensor}}}, optional*) - the reference tensor of `input`. If not given, it is obtained by `input.recovery`.

Example:
```python
>>> c = pydec.Composition(torch.randn(2,3), torch.randn(3))
>>> c
"""
composition{
  components:
    tensor([ 1.7350, -0.0939,  0.0576]),
    tensor([-0.0486,  1.6759,  0.1496]),
  residual:
    tensor([ 0.3260,  0.1409, -0.1552])}
"""
>>> pydec.reciprocal(c)
"""
composition{
  components:
    tensor([-2.6446,  0.3866,  7.1325]),
    tensor([ 0.0740, -6.9031, 18.5227]),
  residual:
    tensor([ 3.0674,  7.0968, -6.4453])}
"""
```