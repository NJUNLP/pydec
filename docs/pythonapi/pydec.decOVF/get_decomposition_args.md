# PYDEC.DECOVF.GET_DECOMPOSITION_ARGS
> pydec.decOVF.get_decomposition_args() →  {{python_dict}}

Returns the currently configured arguments of the decomposition algorithm.

Examples:
```python
>>> pydec.decOVF.set_decomposition_args(arg1=2, arg2='foo') 
>>> pydec.decOVF.get_decomposition_args()
"""
{'arg1': 2, 'arg2': 'foo'}
"""
>>> with pydec.decOVF.using_decomposition_args(arg1=3, arg3=True):
...     pydec.decOVF.get_decomposition_args()
"""
{'arg1': 3, 'arg2': 'foo', 'arg3': True}
"""
```