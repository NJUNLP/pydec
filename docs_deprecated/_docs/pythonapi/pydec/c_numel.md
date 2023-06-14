---
title: "c_numel"
description: pydec.c_numel
---
# PYDEC.C_NUMEL
{% include function.html content="pydec.c_numel(input, count_residual=False) -> int" %}

Returns the the total number of elements of all components of the `input` composition.

{% include function.html content="Parameters:" %}

* **input** ({% include doc.html name="Composition" path="pythonapi/pydec.Composition" %}) - the input composition.
* **count_residual** ([bool](https://docs.python.org/3/library/functions.html#bool), optional) - whether the result is to include residual component or not. Default: *False*.

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
