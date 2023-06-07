# PYDEC.TRANSPOSE
> pydec.transpose(input, dim0, dim1) →  {{{pydec_Composition}}}

Returns a composition that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

The resulting `out` composition shares its underlying storage with the input composition (strided composition), so changing the content of one would change the content of the other.



**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim0** (*{{{python_int}}}*) - the first dimension to be transposed.
* **dim1** (*{{{python_int}}}*) - the second dimension to be transposed.

Example:
```python
>>> c = pydec.zeros(2, 3, c_num=4)
>>> c.c_size()
"""
torch.Size([4, 2, 3])
"""
>>> pydec.transpose(c, 0, 1).c_size()
"""
torch.Size([4, 3, 2])
"""
```