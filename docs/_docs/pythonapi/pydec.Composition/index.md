---
title: "pydec.Composition"
description: API for the module pydec.Composition
---

# PYDEC.COMPOSITION

A {% include codelink.html name="pydec.Composition" path="pythonapi/pydec.Composition/#composition-class-reference" %} is a set of components, where each component is a tensor with the same shape.

## Initializing and basic operations
A  composition of the specified shape can be constructed using the {% include codelink.html name="pydec.Composition()" path="pythonapi/pydec.Composition/#composition-class-reference" %} constructor:
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


A  composition can be constructed from a tensor or another composition using the {% include codelink.html name="pydec.Composition()" path="pythonapi/pydec.Composition/#composition-class-reference" %} constructor:
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

The composition slicing behavior is not exactly the same as tensor, please refer to {% include codelink.html name="Composition.__getitem__" path="pythonapi/pydec.Composition/__getitem__" %} and {% include codelink.html name="Composition.__setitem__" path="pythonapi/pydec.Composition/__setitem__" %}.

For more information about indexing, see {% include doc.html name="Indexing, Slicing, Joining, Mutating Ops" path="pythonapi/pydec/#indexing-slicing-joining-mutating-ops" %}

If you want `torch.autograd` to record operations on composition for automatic differentiation. Do not use the `requires_grad` parameter in the constructor of Composition, otherwise the initialization of Composition as a leaf node cannot be completed by assignment. It is recommended to assign the input with gradient to the Composition without gradient.


<div class="alert alert-info" role="info">
<h4 class="alert-heading">NOTE</h4>
To change an existing composition's {% include inlinecode.html content="torch.device" %} and/or {% include inlinecode.html content="torch.dtype" %}, consider using {% include codelink.html name="to()" path="pythonapi/pydec.Composition/to" %} method on the composition.
</div>

## Composition class reference
{% include class.html content="pydec.Composition" %}
There are a few main ways to create a composition, depending on your use case.
* To create a composition with pre-existing data, pass in a tensor or composition as an argument.
* To create a composition with specific shape, pass in the shape of the component and specify the number of components.
* To create a composition with the same shape (and similar types) as another composition, use `pydec.*_like` composition creation ops (see {% include doc.html name="Creation Ops" path="pythonapi/pydec/#creation-ops" %}).

{% include attribute.html content="Composition._component_tensor" %}

This is the data structure inside composition that stores the components. Its first dimension represents the number of components and the subsequent dimensions represent the shape of the components, i.e.,

`Composition._component_tensor.size(0)` is equal to {% include codelink.html name="Composition.numc()" path="pythonapi/pydec.Composition/numc" %}.
`Composition._component_tensor.size()[1:]` is equal to {% include codelink.html name="Composition.size()" path="pythonapi/pydec.Composition/size" %}.

{% include attribute.html content="Composition._residual_tensor" %}

Special component for storing residuals. It has the same shape as the other components within the composition.

| API                                                                                                                 | Description                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| {% include codelink.html name="Composition.is_cuda" path="pythonapi/pydec.Composition/is_cuda" %}                   | Is `True` if the Composition is stored on the GPU, `False` otherwise.                                                                     |
| {% include codelink.html name="Composition.add" path="pythonapi/pydec.Composition/add" %}                           | Add a scalar or tensor or composition to `self` composition.                                                                              |
| {% include codelink.html name="Composition.add_" path="pythonapi/pydec.Composition/add_" %}                         | In-place version of {% include codelink.html name="add()" path="pythonapi/pydec.Composition/add" %}                                       |
| {% include codelink.html name="Composition.all" path="pythonapi/pydec.Composition/all" %}                           | See {% include codelink.html name="pydec.all()" path="pythonapi/pydec/all" %}                                                             |
| {% include codelink.html name="Composition.any" path="pythonapi/pydec.Composition/all" %}                           | See {% include codelink.html name="pydec.any()" path="pythonapi/pydec/any" %}                                                             |
| {% include codelink.html name="Composition.clone" path="pythonapi/pydec.Composition/clone" %}                       | See {% include codelink.html name="pydec.clone()" path="pythonapi/pydec/clone" %}                                                         |
| {% include codelink.html name="Composition.contiguous" path="pythonapi/pydec.Composition/contiguous" %}             | Returns a contiguous in memory composition containing the same data as `self` composition.                                                |
| {% include codelink.html name="Composition.cpu" path="pythonapi/pydec.Composition/cpu" %}                           | Returns a copy of this object in CPU memory.                                                                                              |
| {% include codelink.html name="Composition.cuda" path="pythonapi/pydec.Composition/cuda" %}                         | Returns a copy of this object in CUDA memory.                                                                                             |
| {% include codelink.html name="Composition.diagonal_scatter" path="pythonapi/pydec.Composition/diagonal_scatter" %} | See {% include codelink.html name="pydec.diagonal_scatter()" path="pythonapi/pydec/diagonal_scatter" %}                                   |
| {% include codelink.html name="Composition.dim" path="pythonapi/pydec.Composition/dim" %}                           | Returns the number of dimensions of the individual components of `self` composition.                                                      |
| {% include codelink.html name="Composition.div" path="pythonapi/pydec.Composition/div" %}                           | See {% include codelink.html name="pydec.div()" path="pythonapi/pydec/div" %}                                                             |
| {% include codelink.html name="Composition.div_" path="pythonapi/pydec.Composition/div_" %}                         | In-place version of {% include codelink.html name="div()" path="pythonapi/pydec.Composition/div" %}                                       |
| {% include codelink.html name="Composition.gather" path="pythonapi/pydec.Composition/gather" %}                     | See {% include codelink.html name="pydec.gather()" path="pythonapi/pydec/gather" %}                                                       |
| {% include codelink.html name="Composition.index_fill_" path="pythonapi/pydec.Composition/index_fill_" %}           | Fills the elements of each component of the `self` composition with value `value` by selecting the indices in the order given in `index`. |
| {% include codelink.html name="Composition.index_fill" path="pythonapi/pydec.Composition/index_fill" %}             | Out-of-place version of {% include codelink.html name="index_fill_()" path="pythonapi/pydec.Composition/index_fill_" %}.                  |
| {% include codelink.html name="Composition.index_select" path="pythonapi/pydec.Composition/index_select" %}         | See {% include codelink.html name="pydec.index_select()" path="pythonapi/pydec/index_select" %}                                           |
| {% include codelink.html name="Composition.is_contiguous" path="pythonapi/pydec.Composition/is_contiguous" %}       | Returns True if `self` composition is contiguous in memory in the order specified by memory format.                                       |
| {% include codelink.html name="Composition.masked_scatter_" path="pythonapi/pydec.Composition/masked_scatter_" %}   | Copies elements from `source` into each component of `self` composition at positions where the `mask` is True.                            |
| {% include codelink.html name="Composition.masked_scatter" path="pythonapi/pydec.Composition/masked_scatter" %}     | Out-of-place version of {% include codelink.html name="masked_scatter_()" path="pythonapi/pydec.Composition/masked_scatter_" %}.          |
| {% include codelink.html name="Composition.masked_fill_" path="pythonapi/pydec.Composition/masked_fill_" %}         | Fills elements of each component of `self` composition with value where `mask` is True.                                                   |
| {% include codelink.html name="Composition.masked_select" path="pythonapi/pydec.Composition/masked_select" %}       | See {% include codelink.html name="pydec.masked_select()" path="pythonapi/pydec/masked_select" %}                                         |
| {% include codelink.html name="Composition.mean" path="pythonapi/pydec.Composition/mean" %}                         | See {% include codelink.html name="pydec.mean()" path="pythonapi/pydec/mean" %}                                                           |
| {% include codelink.html name="Composition.mul" path="pythonapi/pydec.Composition/mul" %}                           | See {% include codelink.html name="pydec.mul()" path="pythonapi/pydec/mul" %}                                                             |
| {% include codelink.html name="Composition.mul_" path="pythonapi/pydec.Composition/mul_" %}                         | In-place version of {% include codelink.html name="mul()" path="pythonapi/pydec.Composition/mul" %}.                                      |
| {% include codelink.html name="Composition.numel" path="pythonapi/pydec.Composition/numel" %}                       | See {% include codelink.html name="pydec.numel()" path="pythonapi/pydec/numel" %}                                                         |
| {% include codelink.html name="Composition.c_numel" path="pythonapi/pydec.Composition/c_numel" %}                   | See {% include codelink.html name="pydec.c_numel()" path="pythonapi/pydec/c_numel" %}                                                     |
| {% include codelink.html name="Composition.numc" path="pythonapi/pydec.Composition/numc" %}                         | See {% include codelink.html name="pydec.numc()" path="pythonapi/pydec/numc" %}                                                           |
| {% include codelink.html name="Composition.permute" path="pythonapi/pydec.Composition/permute" %}                   | See {% include codelink.html name="pydec.permute()" path="pythonapi/pydec/permute" %}                                                     |
| {% include codelink.html name="Composition.reshape" path="pythonapi/pydec.Composition/reshape" %}                   | Returns a composition with the same data and number of elements as `self` but with the specified shape of each component.                 |
| {% include codelink.html name="Composition.reshape_as" path="pythonapi/pydec.Composition/reshape_as" %}             | Returns this composition as the same shape as `other`.                                                                                    |
| {% include codelink.html name="Composition.round" path="pythonapi/pydec.Composition/round" %}                       | See {% include codelink.html name="pydec.round()" path="pythonapi/pydec/round" %}                                                         |
| {% include codelink.html name="Composition.round_" path="pythonapi/pydec.Composition/round_" %}                     | In-place version of {% include codelink.html name="round()" path="pythonapi/pydec.Composition/round" %}.                                  |
| {% include codelink.html name="Composition.scatter_" path="pythonapi/pydec.Composition/scatter_" %}                 | Writes all values from the tensor `src` into each component of `self` at the indices specified in the `index` tensor.                     |
| {% include codelink.html name="Composition.scatter" path="pythonapi/pydec.Composition/scatter" %}                   | Out-of-place version of {% include codelink.html name="scatter_()" path="pythonapi/pydec.Composition/scatter_" %}.                        |
| {% include codelink.html name="Composition.select" path="pythonapi/pydec.Composition/select" %}                     | See {% include codelink.html name="pydec.select()" path="pythonapi/pydec/select" %}                                                       |
| {% include codelink.html name="Composition.size" path="pythonapi/pydec.Composition/size" %}                         | Returns the size of the individual components of the `self` composition.                                                                  |
| {% include codelink.html name="Composition.squeeze" path="pythonapi/pydec.Composition/squeeze" %}                   | See {% include codelink.html name="pydec.squeeze()" path="pythonapi/pydec/squeeze" %}                                                     |
| {% include codelink.html name="Composition.squeeze_" path="pythonapi/pydec.Composition/squeeze_" %}                 | In-place version of {% include codelink.html name="squeeze()" path="pythonapi/pydec.Composition/squeeze" %}.                              |
| {% include codelink.html name="Composition.sub" path="pythonapi/pydec.Composition/sub" %}                           | See {% include codelink.html name="pydec.sub()" path="pythonapi/pydec/sub" %}                                                             |
| {% include codelink.html name="Composition.sub_" path="pythonapi/pydec.Composition/sub_" %}                         | In-place version of {% include codelink.html name="sub()" path="pythonapi/pydec.Composition/sub" %}.                                      |
| {% include codelink.html name="Composition.sum" path="pythonapi/pydec.Composition/sum" %}                           | See {% include codelink.html name="pydec.sum()" path="pythonapi/pydec/sum" %}                                                             |
| {% include codelink.html name="Composition.c_sum" path="pythonapi/pydec.Composition/c_sum" %}                       | See {% include codelink.html name="pydec.c_sum()" path="pythonapi/pydec/c_sum" %}                                                         |
| {% include codelink.html name="Composition.to" path="pythonapi/pydec.Composition/to" %}                             | Performs Tensor dtype and/or device conversion.                                                                                           |
| {% include codelink.html name="Composition.transpose" path="pythonapi/pydec.Composition/transpose" %}               | See {% include codelink.html name="pydec.transpose()" path="pythonapi/pydec/transpose" %}                                                 |
| {% include codelink.html name="Composition.transpose_" path="pythonapi/pydec.Composition/transpose_" %}             | In-place version of {% include codelink.html name="transpose()" path="pythonapi/pydec.Composition/transpose" %}.                          |
| {% include codelink.html name="Composition.type" path="pythonapi/pydec.Composition/type" %}                         | Returns the type if *dtype* is not provided, else casts this object to the specified type.                                                |
| {% include codelink.html name="Composition.type_as" path="pythonapi/pydec.Composition/type_as" %}                   | Returns this composition cast to the type of the given tensor or composition.                                                             |
| {% include codelink.html name="Composition.unsqueeze" path="pythonapi/pydec.Composition/unsqueeze" %}               | See {% include codelink.html name="pydec.unsqueeze()" path="pythonapi/pydec/unsqueeze" %}                                                 |
| {% include codelink.html name="Composition.unsqueeze_" path="pythonapi/pydec.Composition/unsqueeze_" %}             | In-place version of {% include codelink.html name="unsqueeze()" path="pythonapi/pydec.Composition/unsqueeze" %}.                          |
| {% include codelink.html name="Composition.view" path="pythonapi/pydec.Composition/view" %}                         | Returns a new composition with the same data as the `self` composition but of a different `shape`.                                        |
| {% include codelink.html name="Composition.view_as" path="pythonapi/pydec.Composition/view_as" %}                   | View this composition as the same shape as `other`.                                                                                       |

