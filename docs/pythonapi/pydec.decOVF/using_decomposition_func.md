# PYDEC.DECOVF.USING_DECOMPOSITION_FUNC
> CLASS pydec.decOVF.using_decomposition_func(name)

Context-manager that specify the decomposition algorithm with `name`. Must be the name of a registered algorithm.

Also functions as a decorator.

**Parameters:**

* **name** (*{{{python_str}}}*) – name of the specified algorithm.


Example:
```python
>>> with pydec.decOVF.using_decomposition_func("scaling"):
...     pydec.decOVF.get_decomposition_name()
"""
'scaling'
"""

>>> @pydec.decOVF.using_decomposition_func('scaling')
... def foo():
...     print(pydec.decOVF.get_decomposition_name())
>>> foo()
"""
scaling
"""
```