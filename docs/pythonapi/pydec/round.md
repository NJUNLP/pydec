# PYDEC.ROUND
> pydec.round(input, *, decimals=0, out=None) →  {{{pydec_Composition}}}

Rounds elements of each component in `input` to the nearest integer.   

If `other` is a tensor or number, then it is added to `input.residual`.

For integer inputs, follows the array-api convention of returning a copy of the input tensor.

?> This function implements the "round half to even" to break ties when a number is equidistant from two integers (e.g. `round(2.5)` is 2). </br></br> When the `decimals` argument is specified the algorithm used is similar to NumPy's around. See [torch.round()](https://pytorch.org/docs/stable/generated/torch.round.html#torch.round) for details.

Supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics), [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc), and integer or float inputs.

<!-- Not tested on complex inputs-->

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **decimals** (*{{{python_int}}}*) - Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> c = pydec.Composition(torch.tensor([[4.7, -2.3, 9.1, -7.7], [-0.5, 0.5, 1.5, 2.5]]))
>>> c
"""
composition{
  components:
    tensor([ 4.7000, -2.3000,  9.1000, -7.7000]),
    tensor([-0.5000,  0.5000,  1.5000,  2.5000]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> # Values equidistant from two integers are rounded towards the
>>> #   the nearest even value (zero is treated as even)
>>> pydec.round(c)
"""
composition{
  components:
    tensor([ 5., -2.,  9., -8.]),
    tensor([-0., 0., 2., 2.]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> c = pydec.Composition(torch.tensor([[0.1234567]]))
>>> # A positive decimals argument rounds to the to that decimal place
>>> pydec.round(c, decimals=3)
"""
composition{
  components:
    tensor([0.1230]),
  residual:
    tensor([0.])}
"""
>>> c = pydec.Composition(torch.tensor([[1200.1234567]]))
>>> # A negative decimals argument rounds to the left of the decimal
>>> pydec.round(c, decimals=-3)
"""
composition{
  components:
    tensor([1000.]),
  residual:
    tensor([0.])}
"""
```
