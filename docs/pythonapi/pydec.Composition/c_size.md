# PYDEC.COMPOSITION.C_SIZE
> Composition.c_size(dim=None) → torch.Size or int

Returns the size of the `self` composition, including component dimension as the first dimension. If `dim` is not specified, the returned value is a `torch.Size`, a subclass of {{{python_tuple}}}. If `dim` is specified, returns an int holding the size of that dimension.

**Parameters:**

* **dim** (*{{{python_int}}}, optional*) – the dimension for which to retrieve the size.

Example:
```python
>>> c = pydec.zeros(2, 3, 4, c_num=5)
>>> c.c_size()
"""
torch.Size([5, 2, 3, 4])
"""
>>> c.c_size(dim=1) 
"""
2
"""
```
