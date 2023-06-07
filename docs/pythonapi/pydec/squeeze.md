# PYDEC.SQUEEZE
> pydec.squeeze(input, dim=None) →  {{{pydec_Composition}}}

Returns a composition with all specified dimensions of `input` of size *1* removed.

For example, if input is of shape: $(A\times 1 \times B \times C \times 1 \times D)$ then the *input.squeeze()* will be of shape: $(A \times B \times C \times D)$.

When `dim` is given, a squeeze operation is done only in the given dimension(s). If input is of shape: $(A\times 1 \times B)$, `squeeze(input, 0)` leaves the composition unchanged, but `squeeze(input, 1)` will squeeze the composition to the shape $(A\times B)$.

?> `pydec.squeeze()` never remove the component dimension, even if the number of components is 1.

?> The returned composition shares the storage with the input composition, so changing the contents of one will change the contents of the other.

!> If the composition has a batch dimension of size 1, then `squeeze(input)` will also remove the batch dimension, which can lead to unexpected errors. Consider specifying only the dims you wish to be squeezed.


**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}}, optional*) - if given, the input will be squeezed only in the specified dimension.

!? Currently, `dim` does not accept tuples of dimensions.

Example:
```python
>>> c = pydec.zeros(2, 1, 2, 1, 2, c_num=1)
>>> c.size()
"""
torch.Size([2, 1, 2, 1, 2])
"""
>>> c.c_size()
"""
torch.Size([1, 2, 1, 2, 1, 2])
"""
>>> y = pydec.squeeze(c)
>>> y.size()
"""
torch.Size([2, 2, 2])
"""
>>> y.c_size()
"""
torch.Size([1, 2, 2, 2])
"""
>>> y = pydec.squeeze(c, 0)
>>> y.size()
"""
torch.Size([2, 1, 2, 1, 2])
"""
>>> y = pydec.squeeze(c, 1)
>>> y.size()
"""
torch.Size([2, 2, 1, 2])
"""
```