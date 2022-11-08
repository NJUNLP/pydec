---
title: "using_bias_decomposition_func"
description: pydec.using_bias_decomposition_func
---
# PYDEC.USING_BIAS_DECOMPOSITION_FUNC

{% include class.html content="pydec.using_bias_decomposition_func(name)" %}

Context-manager that set the bias decomposition algorithm.

Also functions as a decorator. (Make sure to instantiate with parenthesis.)

{% include function.html content="Parameters:" %}

* **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) - Specifies the name of the bias decomposition algorithm used in the context. Must be the name of a registered algorithm.

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
>>> c + 1
"""
composition 0:
tensor([1.])
composition 1:
tensor([1.])
residual:
tensor([1.])
"""
>>> with pydec.using_bias_decomposition_func('abs_decomposition'):
...     c + 1
"""
composition 0:
tensor([1.5000])
composition 1:
tensor([1.5000])
residual:
tensor([0.])
"""
>>> @pydec.using_bias_decomposition_func('abs_decomposition')
... def add1(c):
...     return c + 1
>>> add1(c) 
"""
composition 0:
tensor([1.5000])
composition 1:
tensor([1.5000])
residual:
tensor([0.])
"""
```