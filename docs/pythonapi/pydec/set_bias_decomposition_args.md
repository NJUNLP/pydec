---
title: "set_bias_decomposition_args"
description: pydec.set_bias_decomposition_args
---
# PYDEC.SET_BIAS_DECOMPOSITION_ARGS

{% include function.html content="pydec.set_bias_decomposition_args(update=True, **kwargs)" %}

Set the default arguments of the bias decomposition algorithm.

{% include function.html content="Parameters:" %}

* **update** ([bool](https://docs.python.org/3/library/functions.html#bool), optional) - If **True**, set the arguments by dictionary update. Otherwise, the previously set arguments are discarded. Default: **False**.

{% include function.html content="Keyword Arguments:" %}

Any number of keyword parameters to set. Do not use the keyword named `update`.

Examples:
```python
>>> pydec.set_bias_decomposition_args(arg1=2, arg2='foo') 
>>> pydec.get_bias_decomposition_args()
"""
{'arg1': 2, 'arg2': 'foo'}
"""
>>> pydec.set_bias_decomposition_args(arg1=4, arg3=True)  
>>> pydec.get_bias_decomposition_args()
"""
{'arg1': 4, 'arg2': 'foo', 'arg3': True}
"""
>>> pydec.set_bias_decomposition_args(update=False, arg1=0) 
>>> pydec.get_bias_decomposition_args()
"""
{'arg1': 0}
"""
```