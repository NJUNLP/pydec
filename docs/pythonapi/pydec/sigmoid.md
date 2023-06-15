# PYDEC.SIGMOID
> pydec.sigmoid(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.sigmoid()](https://pytorch.org/docs/stable/generated/torch.sigmoid.html#torch.sigmoid).

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
    tensor([-1.1610,  0.2827,  0.1398]),
    tensor([-0.6565,  1.0422,  0.2355]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.sigmoid(c)
"""
composition{
  components:
    tensor([-0.2301,  0.0619,  0.0345]),
    tensor([-0.1301,  0.2281,  0.0582]),
  residual:
    tensor([0.5000, 0.5000, 0.5000])}
"""
```