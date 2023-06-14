# PYDEC.DECOVF._NONE_DECOMPOSITION
> pydec.decOVF._none_decomposition(input, func, *, ref=None, inplace=False) -> {{{pydec_Composition}}}:

No decomposition is performed. All components are zeroed and the output of OVF is assigned to the residual.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **func** (*{{{python_callable}}}*) - the nonlinear OVF, should be an element-wise function whose input and output are both tensor.

**Keyword Arguments:**

* **ref** (*{{{torch_Tensor}}}, optional*) - the reference tensor of `input`. If not given, it is obtained by `input.recovery`.
* **inplace** (*{{{python_bool}}}, optional*) - If set to *True*, will do this operation in-place. Default: *False*.

Examples:
```python
>>> c = pydec.Composition(torch.randn(2, 3))
>>> c
"""
composition{
  components:
    tensor([-0.7238, -0.4226,  0.5387]),
    tensor([ 0.7887, -0.5355, -0.2574]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF._none_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0.0649, 0.0000, 0.2813])}
"""
```