# PYDEC.COMPOSITION.INDEX_FILL_
> Composition.index_fill_(dim, index, value) →  {{{pydec_Composition}}}

Fills the elements of each component (including residual) of the `self` composition with value `value` by selecting the indices in the order given in `index`.

**Parameters:**

* **dim** (*{{{python_int}}}*) – dimension along which to index.
* **index** (*IntTensor or LongTensor*) – indices of `self` composition to fill in.
* **value** (*{{{torch_Tensor}}} or Number*) –  the value to fill with.


Example:
```python
>>> c = pydec.Composition(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
>>> c
"""
composition{
  components:
    tensor([1, 2, 3]),
    tensor([4, 5, 6]),
    tensor([7, 8, 9]),
  residual:
    tensor([0, 0, 0])}
"""
>>> index = torch.tensor([0, 2])
>>> c.index_fill_(0, index, -1)
"""
composition{
  components:
    tensor([-1,  2, -1]),
    tensor([-1,  5, -1]),
    tensor([-1,  8, -1]),
  residual:
    tensor([-1,  0, -1])}
"""