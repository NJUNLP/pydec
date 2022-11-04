---
title: "pydec.Composition"
description: API for the module pydec.Composition
---

# pydec.Composition

A `pydec.Composition` is a set of components, where each component is a tensor with the same size.

## Initializing and basic operations
A  composition of the specified size can be constructed using the `pydec.Composition()` constructor:
```python
>>> size = (3, 2)
>>> component_num = 2
>>> c = pydec.Composition(size, component_num)
>>> c
composition 0:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
composition 1:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
residual:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
```

{% include alert.html type="warning" title="WARNING" content="We plan to migrate this interface to `pydec.zeros()`." %}

A  composition can be constructed from a tensor or another composition using the `pydec.Composition()` constructor:
```python
>>> component_num = 2
>>> c_size = (component_num, 3, 2)
>>> t = torch.randn(c_size) # 4 x 3 x 2
>>> pydec.Composition(t)
composition 0:
tensor([[ 0.6367,  0.8254],
        [ 1.0425, -0.1901],
        [ 2.0064,  0.1117]])
composition 1:
tensor([[-0.6671, -1.0998],
        [ 0.1510,  1.4586],
        [-1.0969,  1.2644]])
residual:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
```
{% include alert.html type="info" title="NOTE" content="When you initialize the composition by tensor, each tensor in dimension 0 is treated as the corresponding component." %}

{% include alert.html type="warning" title="WARNING" content="`pydec.Composition()` always copies data. If you have a Composition `data` and just want to change its `requires_grad` flag, use `detach()` to avoid a copy." %}

A composition of specific data type can be constructed by passing a `torch.dtype` and/or a `torch.device` to a constructor or composition creation op:
```python
>>> size = (3, 2)
>>> component_num = 2
>>> pydec.Composition(size, component_num, dtype=torch.int32)
composition 0:
tensor([[0, 0],
        [0, 0],
        [0, 0]], dtype=torch.int32)
composition 1:
tensor([[0, 0],
        [0, 0],
        [0, 0]], dtype=torch.int32)
residual:
tensor([[0, 0],
        [0, 0],
        [0, 0]], dtype=torch.int32)
```

The contents of a conposition can be accessed and modified using Python’s indexing and slicing notation:
```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> c = pydec.Composition(t)
>>> c[:,0]
>>> c[0] = 8
>>> c
```

The composition slicing behavior is not exactly the same as tensor, please refer to {% include codelink.html name="Composition.__getitem__" path="pythonapi/pydec.Composition/__getitem__" %} and {% include codelink.html name="Composition.__setitem__" path="pythonapi/pydec.Composition/__setitem__" %}.

For more information about indexing, see [Indexing, Slicing, Joining, Mutating Ops](pythonapi/pydec/#indexing-slicing-joining-mutating-ops).

If you want `torch.autograd` to record operations on composition for automatic differentiation. Do not use the `requires_grad` parameter in the constructor of Composition, otherwise the initialization of Composition as a leaf node cannot be completed by assignment. It is recommended to assign the input with gradient to the Composition without gradient.


{% include alert.html type="info" title="NOTE" content="To change an existing composition's torch.device and/or torch.dtype, consider using `to()` method on the composition." %}
