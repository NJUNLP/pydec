# PYDEC.PERMUTE
> pydec.permute(input, dims) →  {{{pydec_Composition}}}

Returns a view of the original composition `input` with its dimensions permuted.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dims** (*tuple of int*) - the desired ordering of dimensions.

Example:
```python
>>> c = pydec.zeros(2, 3, 5, c_num=2)
>>> c.size()
"""
torch.Size([2, 3, 5])
"""
>>> c.c_size()
"""
torch.Size([2, 2, 3, 5])
"""
>>> c = pydec.permute(c, (2, 0, 1))
"""
torch.Size([5, 2, 3])
"""
>>> c.c_size()
"""
torch.Size([2, 5, 2, 3])
"""
```