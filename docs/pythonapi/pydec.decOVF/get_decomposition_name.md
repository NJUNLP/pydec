# PYDEC.DECOVF.GET_DECOMPOSITION_NAME
> pydec.decOVF.get_decomposition_name() →  {{python_str}}

Returns the name of the currently enabled decomposition algorithm. 

Examples:
```python
>>> pydec.decOVF.set_decomposition_func("affine")
>>> pydec.decOVF.get_decomposition_name()
"""
'affine'
"""
>>> with pydec.decOVF.using_decomposition_func("scaling"):
...     pydec.decOVF.get_decomposition_name()
"""
'scaling'
"""
```