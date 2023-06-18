# PYDEC.COMPOSITION.VIEW
> Composition.view(*shape) → {{{pydec_Composition}}}

Returns a new composition with the same data as the `self` composition but of a different `shape`.

See [torch.Tensor.view()](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view) for details.

?> `Composition.view()` does not change the shape of the composition dimension.

**Parameters:**

* **shape** (*torch.Size or {{{python_int}}}...*) - Tthe desired size.

Example:
```python
>>> x = pydec.zeros(4, 4, c_num=2)                                     
>>> x.size()
"""
torch.Size([4, 4])
"""
>>> y = x.view(16)
>>> y.size()
"""
torch.Size([16])
"""
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
"""
torch.Size([2, 8])
"""

>>> a = pydec.Composition(torch.randn(5, 1, 2, 3, 4))
>>> a.size()
"""
torch.Size([1, 2, 3, 4])
"""
>>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
>>> b.size()
"""
torch.Size([1, 3, 2, 4])
"""
>>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
>>> c.size()
"""
torch.Size([1, 3, 2, 4])
"""
>>> torch.all(b.eq(c))
"""
tensor(False)
"""
```