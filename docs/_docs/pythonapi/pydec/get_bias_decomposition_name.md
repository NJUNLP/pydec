---
title: "get_bias_decomposition_name"
description: pydec.get_bias_decomposition_name
---
# PYDEC.GET_BIAS_DECOMPOSITION_NAME

{% include function.html content="pydec.get_bias_decomposition_name()" %}

Returns the registered name of the currently enabled bias decomposition function, which may not be the registered name of the default bias decomposition function if the calling is located in the context of {% include codelink.html name="pydec.using_bias_decomposition_func()" path="pythonapi/pydec/using_bias_decomposition_func" %}.

Examples:
```python
>>> pydec.get_bias_decomposition_name()
"""
'none'
"""
>>> pydec.set_bias_decomposition_func('norm_decomposition')
>>> pydec.get_bias_decomposition_name()
"""
'norm_decomposition'
"""
>>> with pydec.using_bias_decomposition_func('abs_decomposition'):
...     pydec.get_bias_decomposition_name()
"""
'abs_decomposition'
"""
```