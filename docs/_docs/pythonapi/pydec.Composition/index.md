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
"""
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
"""
```

<div class="alert alert-warning" role="alert">
<h4 class="alert-heading">WARNING</h4>
We plan to migrate this interface to {% include inlinecode.html content="pydec.zeros()" %}.
</div>


A  composition can be constructed from a tensor or another composition using the `pydec.Composition()` constructor:
```python
>>> component_num = 2
>>> c_size = (component_num, 3, 2)
>>> t = torch.randn(c_size) # 4 x 3 x 2
>>> pydec.Composition(t)
"""
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
"""
```

<div class="alert alert-info" role="info">
<h4 class="alert-heading">NOTE</h4>
When you initialize the composition by tensor, each tensor in dimension <em>0</em> is treated as the corresponding component.
</div>

<div class="alert alert-warning" role="alert">
<h4 class="alert-heading">WARNING</h4>
{% include codelink.html name="pydec.Composition()" path="pythonapi/pydec.Composition/#composition-class-reference" %} always copies data. If you have a Composition {% include inlinecode.html content="data" %} and just want to change its {% include inlinecode.html content="requires_grad" %} flag, use {% include codelink.html name="detach()" path="pythonapi/pydec.Composition/detach" %} to avoid a copy.
</div>

A composition of specific data type can be constructed by passing a `torch.dtype` and/or a `torch.device` to a constructor or composition creation op:
```python
>>> size = (3, 2)
>>> component_num = 2
>>> pydec.Composition(size, component_num, dtype=torch.int32)
"""
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
"""
```

The contents of a conposition can be accessed and modified using Python’s indexing and slicing notation:
```python
>>> t = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> c = pydec.Composition(t)
>>> c[:,0]
"""
composition 0:
tensor(1)
composition 1:
tensor(4)
residual:
tensor(0)
"""
>>> c[0] = 8
>>> c
"""
composition 0:
tensor([8, 8, 8])
composition 1:
tensor([4, 5, 6])
residual:
tensor([0, 0, 0])
"""
```

The composition slicing behavior is not exactly the same as tensor, please refer to {% include codelink.html name="Composition.\_\_getitem\_\_" path="pythonapi/pydec.Composition/__getitem__" %} and {% include codelink.html name="Composition.\_\_setitem\_\_" path="pythonapi/pydec.Composition/__setitem__" %}.

For more information about indexing, see {% include codelink.html name="Indexing, Slicing, Joining, Mutating Ops" path="pythonapi/pydec/#indexing-slicing-joining-mutating-ops" %}

If you want `torch.autograd` to record operations on composition for automatic differentiation. Do not use the `requires_grad` parameter in the constructor of Composition, otherwise the initialization of Composition as a leaf node cannot be completed by assignment. It is recommended to assign the input with gradient to the Composition without gradient.


<div class="alert alert-info" role="info">
<h4 class="alert-heading">NOTE</h4>
To change an existing composition's {% include inlinecode.html content="torch.device" %} and/or {% include inlinecode.html content="torch.dtype" %}, consider using {% include codelink.html name="to()" path="pythonapi/pydec.Composition/to" %} method on the composition.
</div>

## Composition class reference
{% include class.html content="pydec.Composition" %}
There are a few main ways to create a composition, depending on your use case.
* To create a composition with pre-existing data, pass in a tensor or composition as an argument.
* To create a composition with specific size, pass in the size of the component and specify the number of components.
* To create a composition with the same size (and similar types) as another composition, use `pydec.*_like` composition creation ops (see {% include codelink.html name="Creation Ops" path="pythonapi/pydec/#creation-ops" %}).


