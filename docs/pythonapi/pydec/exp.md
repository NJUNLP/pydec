# PYDEC.EXP
> pydec.exp(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.exp()](https://pytorch.org/docs/stable/generated/torch.exp.html#torch.exp).

This is an OVF and the result depends on the currently enabled decomposition algorithm. See {{#auto_link}}pydec.decOVF with_parentheses:false{{/auto_link}} for details.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.
* **ref** (*{{{torch_Tensor}}}, optional*) - the reference tensor of `input`. If not given, it is obtained by `input.recovery`.

Example:
```python
>>> c = pydec.Composition(torch.randn(2,3))
>>> c
"""
composition{
  components:
    tensor([-2.6446,  0.3866,  7.1325]),
    tensor([ 0.0740, -6.9031, 18.5227]),
  residual:
    tensor([ 3.0674,  7.0968, -6.4453])}
"""
>>> pydec.exp(c)
"""
composition{
  components:
    tensor([ 0.7972, -0.2928, -0.4253]),
    tensor([-0.2241, -0.6354, -0.3276]),
  residual:
    tensor([1., 1., 1.])}
"""
```