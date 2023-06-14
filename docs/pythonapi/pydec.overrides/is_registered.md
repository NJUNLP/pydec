# PYDEC.OVERRIDES.IS_REGISTERED
> pydec.overrides.is_registered(torch_function) →  {{python_bool}}

Returns *True* if the `torch_function` is overridden by PyDec. Also returns *True* if the `torch_function` is already registered by {{#auto_link}}pydec.overrides.register_torch_function short with_parentheses:false{{/auto_link}}.

**Parameters:**

* **torch_function** (*{{{python_callable}}}*) - the torch function to check.



Example:
```python
>>> pydec.overrides.is_registered(torch.add)
"""
True
"""
>>> pydec.overrides.is_registered(torch.Tensor.mul)
"""
True
"""
>>> pydec.overrides.is_registered(torch.sin)
"""
False
"""
```