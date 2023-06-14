# PYDEC.DECOVF.SCALING_DECOMPOSITION
> pydec.decOVF.scaling_decomposition(input, func, *, ref=None, inplace=False) -> {{{pydec_Composition}}}:

Mapping `input` to output of OVF based on scaling.

?> This is the decomposition $\bar{\mathscr{D}}$ in our paper, which has consistency.

For a given nonlinear OVF $f$ and input $x$, $x$ is remapped to f(x) by scaling.
$$
f(x)=ax
$$
where $a$ can be obtained by:
$$
a=f(x)/x
$$

This linear transformation is then applied to the composition of $x$ to obtain the composition of the output.

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
    tensor([0.2132, 0.2953, 1.6098]),
    tensor([-0.3280, -0.0111, -1.9587]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF.scaling_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([-0.0000, 0.2953, -0.0000]),
    tensor([ 0.0000, -0.0111,  0.0000]),
  residual:
    tensor([-0., 0., -0.])}
"""

>>> c = pydec.Composition(torch.randn(2, 3), torch.randn(3))
>>> c
"""
composition{
  components:
    tensor([-0.7470,  0.5579,  1.7263]),
    tensor([ 0.8396,  2.2775, -0.4512]),
  residual:
    tensor([ 1.1318, -0.0208,  0.7253])}
"""
>>> pydec.decOVF.scaling_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([-0.7470,  0.5579,  1.7263]),
    tensor([ 0.8396,  2.2775, -0.4512]),
  residual:
    tensor([ 1.1318, -0.0208,  0.7253])}
"""
```