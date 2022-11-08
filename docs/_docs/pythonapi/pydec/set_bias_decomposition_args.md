---
title: "set_bias_decomposition_args"
description: pydec.set_bias_decomposition_args
---
# PYDEC.SET_BIAS_DECOMPOSITION_ARGS

{% include function.html content="pydec.set_bias_decomposition_args(update=True, **kwargs)" %}

Set the default bias decomposition algorithm.

When PyDec is initialized its default bias decomposition algorithm is {% include codelink.html name="none" path="pythonapi/pydec.bias_decomposition/none" %}.

{% include function.html content="Parameters:" %}

* **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) - Specifies the name of the bias decomposition algorithm used by Pydec. Must be the name of a registered algorithm.

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
>>> pydec.set_bias_decomposition_func('abs_decomposition')
>>> c + 1
"""
composition 0:
tensor([1.5000])
composition 1:
tensor([1.5000])
residual:
tensor([0.])
"""
```