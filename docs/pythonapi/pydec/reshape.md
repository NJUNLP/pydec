# PYDEC.RESHAPE
> pydec.reshape(input, shape) →  {{{pydec_Composition}}}

Returns a composition with the same data and number of elements as `input`, but with the specified shape. When possible, the returned tensor will be a view of `input`. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.

See {{#auto_link}}pydec.Composition.view{{/auto_link}} on when it is possible to return a view.

A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the composition to be reshaped.
* **shape** (*tuple of int*) - the new shape.

Example:
```python
>>> c = pydec.zeros(4, c_num=2)
>>> c()[0], c()[1] = torch.arange(4), torch.arange(4, 8)
>>> c
"""
composition{
  components:
    tensor([0., 1., 2., 3.]),
    tensor([4., 5., 6., 7.]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> c = pydec.reshape(c, (2, 2))
"""
composition{
  components:
    tensor([[0., 1.],
            [2., 3.]]),
    tensor([[4., 5.],
            [6., 7.]]),
  residual:
    tensor([[0., 0.],
            [0., 0.]])}
"""
>>> pydec.reshape(c, (-1,))
"""
composition{
  components:
    tensor([0., 1., 2., 3.]),
    tensor([4., 5., 6., 7.]),
  residual:
    tensor([0., 0., 0., 0.])}
“”“
```