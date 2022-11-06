---
title: "zeros_like"
description: pydec.zeros_like
---
# PYDEC.ZEROS_LIKE
{% include function.html content="pydec.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Composition" %}

Returns a composition filled with the scalar value 0, with the same size as `input`. `pydec.zeros_like(input)` is equivalent to `pydec.Composition(input.size(), input.numc())`.

{% include function.html content="Parameters:" %}

* **input** ({% include doc.html name="Composition" path="pythonapi/pydec.Composition" %}) - the size of `input` will determine size of the output composition.

{% include function.html content="Keyword Arguments:" %}

* **dtype** ([torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) - the desired data type of returned composition. Default: if *None*, defaults to the dtype of input.
* **layout** ([torch.layout](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – the desired layout of returned composition. Default: if *None*, defaults to the layout of *input*.
* **device** ([torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) - the desired device of returned composition. Default: if *None*, defaults to the device of input.
* **requires_grad** (bool, optional) - If autograd should record operations on the returned composition. Default: *False*.
* **memory_format** ([torch.memory_format](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format), optional) – the desired memory format of returned Tensor. Default: *torch.preserve_format*.

Example:
```python
>>> input = pydec.Composition((2, 3), component_num=1)
>>> pydec.zeros_like(input)
"""
composition 0:        
tensor([[0., 0., 0.], 
        [0., 0., 0.]])
residual:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
"""
```
