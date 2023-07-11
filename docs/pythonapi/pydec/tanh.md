# PYDEC.TANH
> pydec.tanh(input, *, out=None, ref=None) →  {{{pydec_Composition}}}

See [torch.tanh()](https://pytorch.org/docs/stable/generated/torch.tanh.html#torch.tanh).

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
    tensor([-1.0010, -2.4940,  1.5755]),
    tensor([0.5206, 0.4929, 3.0200]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.tanh(c)
"""
composition{
  components:
    tensor([-0.9305, -1.2016,  0.3428]),
    tensor([0.4840, 0.2375, 0.6570]),
  residual:
    tensor([0., 0., 0.])}
"""
```