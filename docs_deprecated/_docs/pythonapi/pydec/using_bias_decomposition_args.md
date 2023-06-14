---
title: "using_bias_decomposition_args"
description: pydec.using_bias_decomposition_args
---
# PYDEC.USING_BIAS_DECOMPOSITION_ARGS

{% include class.html content="pydec.using_bias_decomposition_args(update=True, **kwargs)" %}

Context-manager that set the arguments of the bias decomposition algorithm.

{% include function.html content="Parameters:" %}

* **update** ([bool](https://docs.python.org/3/library/functions.html#bool), optional) - If *true*, set the arguments by dictionary updating. Otherwise, set the arguments by replacing. Default: *False*.

{% include function.html content="Keyword Arguments:" %}

Any number of keyword parameters to set. Do not use the keyword named `update`.

Examples:
```python
>>> pydec.set_bias_decomposition_args(arg1=2, arg2='foo') 
>>> print(pydec.get_bias_decomposition_args())
"""
{'arg1': 2, 'arg2': 'foo'}
"""
>>> with pydec.using_bias_decomposition_args(arg1=4, arg3=True):
...     print(pydec.get_bias_decomposition_args())"""
{'arg1': 4, 'arg2': 'foo', 'arg3': True}
"""
>>> print(pydec.get_bias_decomposition_args())
"""
{'arg1': 2, 'arg2': 'foo'}
"""
```