# PYDEC.C_NUMEL
> pydec.c_numel(input, count_residual=False) →  {{python_int}}

Returns the the total number of elements of all components of the `input` composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) - the input composition.
* **count_residual** (*{{{python_bool}}}, optional*) - whether the result is to include residual component or not. Default: **False**.

Example:
```python
>>> t = torch.zeros(3, 4, 4)
>>> a = pydec.Composition(t)
>>> pydec.numel(a)
"""
48
"""
>>> pydec.numel(a, count_residual=True)
"""
64
"""
```
