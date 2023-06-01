# PYDEC.INDEX_SELECT
> pydec.index_select(input, dim, index, *, out=None) →  {{{pydec_Composition}}}

Returns a new composition which indexes the `input` composition along dimension `dim` using the entries in `index` which is a *LongTensor*.

The returned composition has the same number of dimensions as the original composition (`input`). The `dim`th dimension has the same size as the length of `index`; other dimensions have the same size as in the original composition.

?> The returned composition does **not** use the same storage as the original composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}}*) – the dimension in which we index.
* **index** (*IntTensor or LongTensor*) – the 1-D tensor containing the indices to index.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> c = pydec.Composition(torch.rand((2, 3, 4)))
>>> c
"""
composition{
  components:
    tensor([[0.4455, 0.8479, 0.4028, 0.2596],
            [0.8806, 0.3307, 0.3219, 0.8061],
            [0.2695, 0.4041, 0.2377, 0.2028]]),
    tensor([[0.0476, 0.3166, 0.5378, 0.2456],
            [0.2425, 0.3052, 0.6299, 0.6326],
            [0.4829, 0.7364, 0.4053, 0.5290]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> indices = torch.tensor([0, 2])
>>> pydec.index_select(c, 0, indices)
"""
composition{
  components:
    tensor([[0.4455, 0.8479, 0.4028, 0.2596],
            [0.2695, 0.4041, 0.2377, 0.2028]]),
    tensor([[0.0476, 0.3166, 0.5378, 0.2456],
            [0.4829, 0.7364, 0.4053, 0.5290]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> pydec.index_select(c, 1, indices)
"""
composition{
  components:
    tensor([[0.4455, 0.4028],
            [0.8806, 0.3219],
            [0.2695, 0.2377]]),
    tensor([[0.0476, 0.5378],
            [0.2425, 0.6299],
            [0.4829, 0.4053]]),
  residual:
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])}
"""
```
