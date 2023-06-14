# PYDEC.DECOVF.GET_DECOMPOSITION_FUNC
> pydec.decOVF.get_decomposition_func() →  {{python_callable}}

Returns the function of the currently enabled decomposition algorithm.

Examples:
```python
>>> pydec.decOVF.set_decomposition_func("affine")
>>> pydec.decOVF.get_decomposition_func() == pydec.decOVF.affine_decomposition
"""
True
"""
>>> with pydec.decOVF.using_decomposition_func("scaling"):
...     pydec.decOVF.get_decomposition_func()
"""
<function scaling_decomposition at 0x0000018F72D78E50>
"""
```