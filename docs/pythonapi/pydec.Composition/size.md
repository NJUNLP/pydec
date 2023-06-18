# PYDEC.COMPOSITION.SIZE
> Composition.size(dim=None) → torch.Size or int

Returns the size of the `self` composition. If `dim` is not specified, the returned value is a `torch.Size`, a subclass of {{{python_tuple}}}. If `dim` is specified, returns an int holding the size of that dimension.

?> To obtain the size including the component dimension, use {{#auto_link}}pydec.Composition.c_size{{/auto_link}}.

**Parameters:**

* **dim** (*{{{python_int}}}, optional*) – the dimension for which to retrieve the size.

Example:
```python
>>> c = pydec.zeros(2, 3, 4, c_num=5)
>>> c.size()
"""
torch.Size([2, 3, 4])
"""
>>> c.size(dim=1)
"""
3
"""
```
