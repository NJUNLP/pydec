# PYDEC.DECOVF.SET_DECOMPOSITION_ARGS
> pydec.decOVF.SET_decomposition_args(update=True, **kwargs)

Specify the arguments of the decomposition algorithm globally.

?> It can be arguments that are not accepted by the currently enabled decomposition function, in which case they are not passed into the function.

**Parameters:**

* **update** (*{{{python_bool}}}, optional*) - If *True*, set the arguments by dictionary update. Otherwise, the previously set arguments are discarded. Default: *True*.

**Keyword Arguments:**

Any number of keyword arguments, except `update`.


Examples:
```python
>>> pydec.decOVF.set_decomposition_args(arg1=2, arg2='foo') 
>>> pydec.decOVF.get_decomposition_args()
"""
{'arg1': 2, 'arg2': 'foo'}
"""
>>> pydec.decOVF.set_decomposition_args(arg1=3, arg3=True)
>>> pydec.decOVF.get_decomposition_args()
"""
{'arg1': 3, 'arg2': 'foo', 'arg3': True}
"""
>>> pydec.decOVF.set_decomposition_args(update=False, arg1=0)
>>> pydec.decOVF.get_decomposition_args()
"""
{'arg1': 0}
"""
```