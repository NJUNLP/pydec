# PYDEC.DIAGONAL_SCATTER
> pydec.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1) →  {{{pydec_Composition}}}

Embeds the values of the `src` tensor into `input` composition along the diagonal elements of every component in `input`, with respect to `dim1` and `dim2`.

This function returns a composition with fresh storage; it does not return a view.

The argument `offset` controls which diagonal to consider:

* If `offset` = 0, it is the main diagonal.
* If `offset` > 0, it is above the main diagonal.
* If `offset` < 0, it is below the main diagonal.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the composition to be reshaped. Must be at least 2-dimensional.
* **src** (*{{{torch_Tensor}}}*) - the tensor to embed into `input`.
* **offset** (*{{{python_int}}}, optional*) - which diagonal to consider. Default: 0 (main diagonal).
* **dim1** (*{{{python_int}}}, optional*) - first dimension with respect to which to take diagonal. Default: 0.
* **dim2** (*{{{python_int}}}, optional*) - second dimension with respect to which to take diagonal. Default: 1.

<!-- TODO: add pydec.diagonal API -->

!> This API may change in the near future.

Examples:
```python
>>> a = pydec.zeros(3, 3, c_num=2)
>>> a
"""
composition{
  components:
    tensor([[0., 0., 0.],  
            [0., 0., 0.],  
            [0., 0., 0.]]),
    tensor([[0., 0., 0.],  
            [0., 0., 0.],  
            [0., 0., 0.]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])}
"""
>>> pydec.diagonal_scatter(a, torch.ones(3), 0)
"""
composition{
  components:
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]),
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]),
  residual:
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])}
"""
>>> pydec.diagonal_scatter(a, torch.ones(2), 1)
"""
composition{
  components:
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]]),
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]]),
  residual:
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]])}
"""
```