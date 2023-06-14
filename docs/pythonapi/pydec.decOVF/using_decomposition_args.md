# PYDEC.DECOVF.USING_DECOMPOSITION_ARGS
> CLASS pydec.decOVF.using_decomposition_ARGS(update=True, **kwargs)

Context-manager that specify the arguments of the decomposition algorithm.

?> It can be arguments that are not accepted by the currently enabled decomposition function, in which case they are not passed into the function.


Also functions as a decorator.

**Parameters:**

* **update** (*{{{python_bool}}}, optional*) - If *True*, set the arguments by dictionary update. Otherwise, the previously set arguments are discarded. Default: *True*.

**Keyword Arguments:**

Any number of keyword arguments, except `update`.

Example:
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
>>> with pydec.decOVF.using_decomposition_args(update=False, arg1=0):
...     pydec.decOVF.get_decomposition_args()
"""
{'arg1': 0}
"""
>>> pydec.decOVF.get_decomposition_args()
"""
{'arg1': 2, 'arg2': 'foo'}
"""
```