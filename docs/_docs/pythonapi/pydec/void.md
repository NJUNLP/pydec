---
title: "void"
description: pydec.void
---
# PYDEC.VOID
{% include function.html content="pydec.void(*, dtype=None, device=None, requires_grad=False) -> Composition" %}

Returns an empty composition.

{% include function.html content="Keyword Arguments:" %}

* **dtype** ([torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) - the desired data type of returned composition. Default: if *None*, uses a global default (see [torch.set_default_tensor_type()](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html#torch.set_default_tensor_type)).
* **device** ([torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) - the desired device of returned composition. Default: if *None*, uses the current device for the default tensor type (see [torch.set_default_tensor_type()](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html#torch.set_default_tensor_type)). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ([bool](https://docs.python.org/3/library/functions.html#bool), optional) - If autograd should record operations on the returned composition. Default: *False*.

Example:
```python
>>> pydec.void()
"""
residual: 
tensor(0.)
"""
```
