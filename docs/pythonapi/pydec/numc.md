# PYDEC.NUMC
> pydec.numc(input) →  {{python_int}}

Returns the number of components in the `input` composition. The residual component will not be counted.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) - the input composition.

Example:
```python
>>> a = pydec.zeros((1, 2, 3, 4, 5), component_num=4)
>>> pydec.numc(a)
"""
4
"""
>>> t = torch.zeros(3, 4, 4)
>>> a = pydec.Composition(t)
>>> pydec.numc(a)
"""
3
"""
```
