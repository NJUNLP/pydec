# PYDEC.DECOVF.SET_DECOMPOSITION_FUNC
> pydec.decOVF.set_decomposition_func()

Specify the decomposition algorithm with `name` globally. Must be the name of a registered algorithm.

When PyDec is initialized its default decomposition algorithm is {{#auto_link}}pydec.decOVF.affine_decomposition{{/auto_link}}.

**Parameters:**

* **name** (*{{{python_str}}}*) – name of the specified algorithm.

**Raises:**
* {{{python_ValueError}}} - if the `name` is not registered.


Examples:
```python
>>> pydec.decOVF.set_decomposition_func("affine")
>>> pydec.decOVF.get_decomposition_name()
"""
'affine'
"""
>>> pydec.decOVF.set_decomposition_func("scaling")
>>> pydec.decOVF.get_decomposition_name()
"""
'scaling'
"""
```