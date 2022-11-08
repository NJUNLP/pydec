---
title: "get_bias_decomposition_func"
description: pydec.get_bias_decomposition_func
---
# PYDEC.GET_BIAS_DECOMPOSITION_FUNC

{% include function.html content="pydec.get_bias_decomposition_func()" %}

Returns the currently enabled bias decomposition function, which may not be the default bias decomposition function if the calling is located in the context of {% include codelink.html name="pydec.using_bias_decomposition_func()" path="pythonapi/pydec/using_bias_decomposition_func" %}.

Examples:
```python
>>> pydec.get_bias_decomposition_func().__name__
"""
'_none_decomposition'
"""
>>> pydec.set_bias_decomposition_func('norm_decomposition') 
>>> pydec.get_bias_decomposition_func().__name__
"""
'norm_decomposition'
"""
>>> with pydec.using_bias_decomposition_func('abs_decomposition'):
...     pydec.get_bias_decomposition_func().__name__
"""
'abs_decomposition'
"""
```