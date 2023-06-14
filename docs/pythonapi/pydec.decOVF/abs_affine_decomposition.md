# PYDEC.DECOVF.ABS_AFFINE_DECOMPOSITION
> pydec.decOVF.abs_affine_decomposition(input, func, *, ref=None, inplace=False) -> {{{pydec_Composition}}}:

Mapping `input` to output of OVF based on affine transformation. All components in the `input` are first taken in absolute value. 

?> This is the decomposition $\hat{\mathscr{D}}$(abs) in our paper, which is not consistent.

For a given nonlinear OVF $f$ and input $x$. Let the components of x be $c_1,\cdots,c_m$, i.e., $x=\sum_i^m c_i$. After taking the absolute value for each component, we have
$$
x^\prime = \sum_i^m |c_i|
$$
$x^\prime$ is remapped to f(x) using the affine transformation.

$$
f(x)=ax^\prime+b
$$
where $a$ and $b$ can be obtained by:
$$
b=f(0)\\
a=[f(x)-b]/x^\prime
$$

This affine transformation is then applied to the composition of $x^\prime$ to obtain the composition of the output.

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
    tensor([ 1.3503, -0.0398,  1.0369]),
    tensor([-0.5818, -0.4043, -1.4794]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF.abs_affine_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([0.5370, 0.0000, 0.0000]),
    tensor([0.2314, 0.0000, 0.0000]),
  residual:
    tensor([0., 0., 0.])}
"""

>>> c = pydec.Composition(torch.randn(2, 3), torch.randn(3))
>>> c
"""
composition{
  components:
    tensor([-0.2730,  1.0810,  0.4577]),
    tensor([-1.9134, -0.8740, -1.1139]),
  residual:
    tensor([0.0365, 0.1602, 0.0418])}
"""
>>> pydec.decOVF.abs_affine_decomposition(c, torch.nn.functional.relu)
"""
composition{
  components:
    tensor([-0.0046,  0.1145, -0.0122]),
    tensor([-0.0319,  0.0926, -0.0296]),
  residual:
    tensor([0.0365, 0.1602, 0.0418])}
"""
```