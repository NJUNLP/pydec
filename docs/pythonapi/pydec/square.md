# PYDEC.SQUARE
> pydec.square(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.square()](https://pytorch.org/docs/stable/generated/torch.square.html#torch.square).

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
    tensor([ 0.5997,  2.5464, -0.1751]),
    tensor([ 0.3233,  0.8692, -1.3680]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.square(c)
"""
composition{
  components:
    tensor([0.5535, 8.6972, 0.2702]),
    tensor([0.2984, 2.9686, 2.1109]),
  residual:
    tensor([0., 0., 0.])}
"""
```