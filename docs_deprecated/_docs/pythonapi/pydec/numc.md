---
title: "numc"
description: pydec.numc
---
# PYDEC.NUMC
{% include function.html content="pydec.numc(input) -> int" %}

Returns the number of components in the `input` composition. The residual component will not be counted.

{% include function.html content="Parameters:" %}

* **input** ({% include doc.html name="Composition" path="pythonapi/pydec.Composition" %}) - the input composition.

Example:
```python
>>> a = pydec.Composition((1, 2, 3, 4, 5), component_num=4)
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
