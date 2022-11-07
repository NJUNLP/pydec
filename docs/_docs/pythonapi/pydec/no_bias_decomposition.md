---
title: "no_bias_decomposition"
description: pydec.no_bias_decomposition
---
# PYDEC.NO_BIAS_DECOMPOSITION

{% include function.html content="pydec.using_bias_decomposition_func(name)" %}

Context-manager that disable bias decomposition.

Also functions as a decorator. (Make sure to instantiate with parenthesis.)

Examples:
```python
>>> c = pydec.Composition((1,), 2) 
>>> c[:]=1.0
>>> c
"""
composition 0:
tensor([1.])  
composition 1:
tensor([1.])  
residual:     
tensor([0.])
"""
>>> pydec.set_bias_decomposition_func("abs_decomposition")
>>> c + 1
"""
composition 0:
tensor([1.5000])
composition 1:
tensor([1.5000])
residual:
tensor([0.])
"""
>>> with pydec.no_bias_decomposition():
...     c + 1
"""
composition 0:
tensor([1.])
composition 1:
tensor([1.])
residual:
tensor([1.])
"""
>>> @pydec.no_bias_decomposition()
... def add1(c):
...     return c + 1
>>> add1(c) 
"""
composition 0:
tensor([1.])
composition 1:
tensor([1.])
residual:
tensor([1.])
"""
```