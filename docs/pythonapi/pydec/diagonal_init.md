# PYDEC.DIAGONAL_INIT
> pydec.diagonal_init(input, src, dim, offset=0) →  {{{pydec_Composition}}}

Embeds the values of the `src` tensor into `input` composition along the diagonal components of `input`, with respect to `dim`.
This function is similar to {{#auto_link}}pydec.diagonal_scatter{{/auto_link}}, except that one dimension argument is specified as the component dimension.

This function returns a composition with fresh storage; it does not return a view.

The argument `offset` controls which diagonal to consider:

* If `offset` = 0, it is the main diagonal.
* If `offset` > 0, it is above the main diagonal.
* If `offset` < 0, it is below the main diagonal.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the composition to be reshaped. Must be at least 2-dimensional.
* **src** (*{{{torch_Tensor}}}*) - the tensor to embed into `input`.
* **dim** (*{{{python_int}}}*) - the dimension with respect to which to take diagonal.
* **offset** (*{{{python_int}}}, optional*) - which diagonal to consider. Default: 0 (main diagonal).

Examples:
```python
>>> x = torch.randn(2, 3)
>>> x
"""
tensor([[ 0.2022,  1.2868, -1.3549],
        [ 0.8925,  0.8642, -0.6798]])
"""
>>> c = pydec.zeros(2, 3, c_num=3)
>>> pydec.diagonal_init(c, src=x, dim=1)
"""
composition{
  components:
    tensor([[0.2022, 0.0000, 0.0000],
            [0.8925, 0.0000, 0.0000]]),
    tensor([[0.0000, 1.2868, 0.0000],
            [0.0000, 0.8642, 0.0000]]),
    tensor([[ 0.0000,  0.0000, -1.3549],
            [ 0.0000,  0.0000, -0.6798]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
>>> x = torch.randn(2, 2)
"""
tensor([[ 0.9815,  0.1704],
        [-1.0634,  0.4566]])
"""
>>> pydec.diagonal_init(c, src=x, dim=1, offset=-1)
"""
composition{
  components:
    tensor([[0., 0., 0.],
            [0., 0., 0.]]),
    tensor([[ 0.9815,  0.0000,  0.0000],
            [-1.0634,  0.0000,  0.0000]]),
    tensor([[0.0000, 0.1704, 0.0000],
            [0.0000, 0.4566, 0.0000]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
```