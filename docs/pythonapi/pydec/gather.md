# PYDEC.GATHER
> pydec.gather(input, dim, index, *, sparse_grad=False, out=None) →  {{{pydec_Composition}}}

Gathers values along an axis specified by *dim*.

For a 3-D composition the output is specified by:
```python
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

See [torch.gather()](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather) to understand this function.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the source composition.
* **dim** (*{{{python_int}}}*) – the axis along which to index.
* **index** (*LongTensor*) – the indices of elements to gather.

**Keyword Arguments:**
* **sparse_grad** (*{{{python_bool}}}, optional*) - If **True**, gradient w.r.t. **input** will be a sparse tensor.
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> c = pydec.zeros((2, 2), 2)
>>> c()[0] = torch.tensor([[1, 2], [3, 4]])
>>> c()[1] = torch.tensor([[5, 6], [7, 8]])
>>> c
"""
composition{
  components:
    tensor([[1., 2.],
            [3., 4.]]),
    tensor([[5., 6.],
            [7., 8.]]),
  residual:
    tensor([[0., 0.],
            [0., 0.]])}
"""
>>> pydec.gather(c, 1, torch.tensor([[0, 0], [1, 0]]))
"""
composition{
  components:
    tensor([[1., 1.],
            [4., 3.]]),
    tensor([[5., 5.],
            [8., 7.]]),
  residual:
    tensor([[0., 0.],
            [0., 0.]])}
"""
```
