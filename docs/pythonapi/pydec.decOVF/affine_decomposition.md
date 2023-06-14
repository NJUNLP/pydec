# PYDEC.DECOVF.AFFINE_DECOMPOSITION
> pydec.decOVF.affine_decomposition(input, func, *, ref=None, inplace=False) -> {{{pydec_Composition}}}:

Mapping `input` to output of OVF based on affine transformation.

?> This is the decomposition $\hat{\mathscr{D}}$(signed) in our paper, which has consistency.



For a given nonlinear OVF $f$ and input $x$, $x$ is remapped to f(x) using the affine transformation.
$$
f(x)=ax+b
$$
where $a$ and $b$ can be obtained by:
$$
b=f(0)\\
a=[f(x)-b]/x
$$

This affine transformation is then applied to the composition of $x$ to obtain the composition of the output.

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
    tensor([-0.7532, -0.1263,  0.9695]),
    tensor([ 0.2721,  0.8744, -0.4808]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF.affine_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([ 0.0000, -0.1263,  0.9695]),
    tensor([-0.0000,  0.8744, -0.4808]),
  residual:
    tensor([0., 0., 0.])}
"""

>>> c = pydec.Composition(torch.randn(2, 3), torch.randn(3))
>>> c
"""
composition{
  components:
    tensor([ 1.8942, -1.1932,  0.0719]),
    tensor([ 1.0633,  2.1078, -1.4699]),
  residual:
    tensor([-0.1706,  0.5189, -0.0086])}
"""
>>> pydec.decOVF.affine_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([ 1.7849, -1.1932, -0.0000]),
    tensor([1.0019, 2.1078, 0.0000]),
  residual:
    tensor([0.0000, 0.5189, 0.0000])}
"""
```