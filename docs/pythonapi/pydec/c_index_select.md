# PYDEC.C_INDEX_SELECT
> pydec.c_index_select(input, index, with_residual=True, *, out=None) →  {{{pydec_Composition}}}

Returns a new composition which indexes the `input` composition along the component dimension using the entries in `index` which is a *LongTensor*.

The returned composition has the same shape as the original composition (`input`). The component dimension has the same size as the length of `index`.

?> The returned composition does **not** use the same storage as the original composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **index** (*IntTensor or LongTensor*) – the 1-D tensor containing the indices to index.
* **with_residual** (*{{{python_bool}}}*) -Whether to index the residual at the same time or not. Default: **True**.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> c = pydec.Composition(torch.rand((3, 2, 4)))
>>> c
"""
composition{
  components:
    tensor([[0.3565, 0.0307, 0.0570, 0.9813],
            [0.8384, 0.2908, 0.8036, 0.8566]]),
    tensor([[0.9873, 0.5656, 0.5307, 0.8458],
            [0.3855, 0.2060, 0.2321, 0.2708]]),
    tensor([[0.0540, 0.9807, 0.8475, 0.5492],
            [0.9125, 0.6500, 0.0476, 0.8215]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> indices = torch.tensor([0, 2])
>>> pydec.c_index_select(c, indices)
"""
composition{
  components:
    tensor([[0.3565, 0.0307, 0.0570, 0.9813],
            [0.8384, 0.2908, 0.8036, 0.8566]]),
    tensor([[0.0540, 0.9807, 0.8475, 0.5492],
            [0.9125, 0.6500, 0.0476, 0.8215]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
```
