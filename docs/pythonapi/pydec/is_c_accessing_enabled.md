# PYDEC.IS_C_ACCESSING_ENABLED
> pydec.is_c_accessing_enabled  →  {{{python_bool}}}

Returns *True* if component accessing mode is currently enabled.


Example:
```python
>>> pydec.is_c_accessing_enabled()
"""
False
"""
>>> with pydec.enable_c_accessing():
...     pydec.is_c_accessing_enabled()
"""
True
"""
```