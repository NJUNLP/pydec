# PYDEC.DECOVF.HYBRID_AFFINE_DECOMPOSITION
> pydec.decOVF.hybrid_affine_decomposition(input, func, *, ref=None, threshold=0, inplace=False) -> {{{pydec_Composition}}}:

Mapping `input` to output of OVF based on affine transformation. Use `threshold` to control whether to take absolute values for components.

?> Consistency is not guaranteed due to the introduction of the {{#auto_link}}pydec.decOVF.abs_affine_decomposition short with_parentheses:false{{/auto_link}} algorithm.

For a given nonlinear OVF $f$ and input $x$. Let the components of x be $c_1,\cdots,c_m$, i.e., $x=\sum_i^m c_i$. There is an indicator $r\in [0,1]$ to measure whether these components are antagonistic to each other.
$$
r=\frac{|\sum_i^m c_i|}{\sum_i^m |c_i|}
$$

Once $r<$`threshold`, take absolute values for $c_1,\cdots,c_m$ to construct $x^\prime$. Then run the {{#auto_link}}pydec.decOVF.abs_affine_decomposition short with_parentheses:false{{/auto_link}} algorithm, otherwise run the {{#auto_link}}pydec.decOVF.affine_decomposition short with_parentheses:false{{/auto_link}} algorithm.

!> For each element in the input tensor of `func`, the result of decomposition is either the output of {{#auto_link}}pydec.decOVF.affine_decomposition short with_parentheses:false{{/auto_link}} or the output of {{#auto_link}}pydec.decOVF.abs_affine_decomposition short with_parentheses:false{{/auto_link}}, rather than their interpolation.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **func** (*{{{python_callable}}}*) - the nonlinear OVF, should be an element-wise function whose input and output are both tensor.

**Keyword Arguments:**

* **ref** (*{{{torch_Tensor}}}, optional*) - the reference tensor of `input`. If not given, it is obtained by `input.recovery`.
* **threshold** (*{{{python_float}}}, optional*) - the threshold to control whether to use the {{#auto_link}}pydec.decOVF.abs_affine_decomposition short with_parentheses:false{{/auto_link}} algorithm.
* **inplace** (*{{{python_bool}}}, optional*) - If set to *True*, will do this operation in-place. Default: *False*.

Examples:
```python
>>> c = pydec.Composition(torch.randn(2, 3))
>>> c
"""
composition{
  components:
    tensor([ 0.9953, -0.1365,  2.6023]),
    tensor([ 1.7294,  1.1527, -0.3457]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF.hybrid_affine_decomposition(c, torch.nn.functional.relu, threshold=0)
"""
composition{
  components:
    tensor([ 0.9953, -0.1365,  2.6023]),
    tensor([ 1.7294,  1.1527, -0.3457]),
  residual:
    tensor([0., 0., 0.])}
"""
>>> pydec.decOVF.hybrid_affine_decomposition(c, torch.nn.functional.relu, threshold=0.8)
"""
composition{
  components:
    tensor([0.9953, 0.1076, 1.9920]),
    tensor([1.7294, 0.9086, 0.2646]),
  residual:
    tensor([0., 0., 0.])}
"""
```