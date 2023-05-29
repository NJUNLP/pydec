---
title: "get_bias_decomposition_args"
description: pydec.get_bias_decomposition_args
---
# PYDEC.GET_BIAS_DECOMPOSITION_ARGS

{% include function.html content="pydec.get_bias_decomposition_args() -> Dict[str, Any]:" %}

Returns the currently enabled arguments of the bias decomposition function, which may not be the default arguments if the calling is located in the context of {% include codelink.html name="pydec.using_bias_decomposition_args()" path="pythonapi/pydec/using_bias_decomposition_args" %}.

Examples:
```python
>>> pydec.set_bias_decomposition_args(arg1=2, arg2='foo') 
>>> print(pydec.get_bias_decomposition_args())
"""
{'arg1': 2, 'arg2': 'foo'}
"""
>>> with pydec.using_bias_decomposition_args(arg1=3):
...     print(pydec.get_bias_decomposition_args())
"""
{'arg1': 3, 'arg2': 'foo'}
"""
```