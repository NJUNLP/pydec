# PYDEC.COMPOSITION

A {{#auto_link}}pydec.Composition with_parentheses:false{{/auto_link}} is a set of components, where each component is a tensor with the same shape.

## Initializing and basic operations
A composition can be constructed from a tensor or another composition using the {{#auto_link}}pydec.Composition{{/auto_link}} constructor:
```python
>>> pydec.Composition(torch.rand(3, 2, 2))
>>> c
"""
composition{
  components:
    tensor([[0.6181, 0.6700],
            [0.0326, 0.0473]]),
    tensor([[0.0296, 0.4895],
            [0.6644, 0.2424]]),
    tensor([[0.3879, 0.3579],
            [0.1515, 0.8610]]),
  residual:
    tensor([[0., 0.],
            [0., 0.]])}
"""

>>> pydec.Composition(torch.zeros(3, 1, 2), torch.rand(1, 2))
"""
composition{
  components:
    tensor([[0., 0.]]),
    tensor([[0., 0.]]),
    tensor([[0., 0.]]),
  residual:
    tensor([[0.7175, 0.7405]])}
""" 
```

?> When initializing the composition by tensors, each tensor in dimension *0* is treated as a component.

!> {{#auto_link}}pydec.Composition{{/auto_link}} always copies **data**. If you have a Composition **data** and just want to change its **requires_grad** flag, use {{#auto_link}}pydec.Composition.requires_grad_ short:2{{/auto_link}} or {{#auto_link}}pydec.Composition.detach short:2{{/auto_link}} to avoid a copy. If you have a tensor and want to avoid a copy, use {{#auto_link}}pydec.as_composition{{/auto_link}}.


A composition of specific data type can be constructed by passing a {{{torch_dtype}}} and/or a {{{torch_device}}} to a constructor or composition creation op:
```python
>>> pydec.zeros((3, 2), 2, dtype=torch.int32)
"""
composition{
  components:
    tensor([[0, 0],
            [0, 0],
            [0, 0]]),
    tensor([[0, 0],
            [0, 0],
            [0, 0]]),
  residual:
    tensor([[0, 0],
            [0, 0],
            [0, 0]]),
  dtype=torch.int32}
"""
```

The contents of a composition can be accessed and modified using Python's indexing and slicing notation:
```python
>>> c = pydec.Composition(torch.tensor([[1, 2, 3], [4, 5, 6]]))
>>> c[0]
"""
composition{
  components:
    tensor(1),
    tensor(4),
  residual:
    tensor(0)}
"""
>>> c[1:] = 8
>>> c
"""
composition{
  components:
    tensor([1, 8, 8]),
    tensor([4, 8, 8]),
  residual:
    tensor([0, 0, 0])}
"""
```

?> If you need to manipulate components of a composition, please refer to [Component Accessing](understanding-composition.md#component-accessing).

For more information about indexing, see [Indexing, Slicing, Joining, Mutating Ops](/pythonapi/pydec/index.md#indexing-slicing-joining-mutating-ops).

!> If you want `torch.autograd` to record operations on composition for automatic differentiation. Do not use the `requires_grad` parameter in the constructor of Composition, otherwise the initialization of Composition as a leaf node cannot be completed by assignment. It is recommended to assign tensors with gradient to the composition without gradient.

?> To change an existing composition's {{{torch_device}}} and/or {{{torch_dtype}}}, consider using {{#auto_link}}pydec.Composition.to short:2{{/auto_link}} method on the composition.

## Composition class reference
> CLASS pydec.Composition
There are a few main ways to create a composition, depending on your use case.
* To create a composition with pre-existing data, pass in a tensor or composition as an argument.
* To create a composition with specific size, use **pydec.zeros()** composition creation ops (see [Creation Ops](/pythonapi/pydec/index.md#creation-ops)).
* To create a composition with the same size (and similar types) as another composition, use **pydec.zeros_like()** composition creation ops (see [Creation Ops](/pythonapi/pydec/index.md#creation-ops)).

Example:
```python
>>> c = pydec.Composition(torch.randn(2, 2, 3))
"""
composition{
  components:
    tensor([[-0.2087,  1.0742,  0.1777],
            [ 0.9413,  0.2100,  0.0274]]),
    tensor([[-1.4433,  0.0802,  0.4976],
            [-1.1147, -0.6400, -1.1477]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
>>> c_copy = pydec.Composition(c) # equal to c.clone().detach()
```
#### Attributes
> Composition.components

This is the data structure inside composition that stores the components. Its first dimension represents the number of components and the subsequent dimensions represent the shape of the components, i.e.,

`Composition.components.size(0)` is the result of `Composition.numc()`.

`Composition.components.size()[1:]` is the result of `Composition.size()`.

?> This is a read-only attribute.

> Composition.residual

A special component for storing residuals. It has the same shape as the other components within the composition.

?> This is a read-only attribute.

> Composition.recovery

The tensor that is decomposed to Composition, obtained by summing up each component.

`Composition.recovery` is the result of `Composition.c_sum()`.


> Composition.requires_grad

Is **True** if gradients need to be computed for this Composition, **False** otherwise.

> Composition.shape

Alias of {{#auto_link}}pydec.Composition.size short:1{{/auto_link}}.

> Composition.device

Is the {{{torch_device}}} where this Composition is.

> Composition.dtype

The data type of this Composition.

> Composition.layout

The memory layout of this Composition.

!> Currently PyDec is not designed for the sparse tensor (**torch.sparse_coo**) and no testing has been done for it.

> Composition.T

Returns a view of this composition with its dimensions reversed.

> Composition.mT

Returns a view of this tensor with the last two dimensions transposed.

| API                                                                                           | Description                                                                                                                               |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.Composition.is_cuda short:1 with_parentheses:false{{/auto_link}}          | Is `True` if the Composition is stored on the GPU, `False` otherwise.                                                                     |
| {{#auto_link}}pydec.Composition.add short:1 with_parentheses:false{{/auto_link}}              | Add a scalar or tensor or composition to `self` composition.                                                                              |
| {{#auto_link}}pydec.Composition.add_ short:1 with_parentheses:false{{/auto_link}}             | In-place version of {{#auto_link}}pydec.Composition.add short:2{{/auto_link}}                                                             |
| {{#auto_link}}pydec.Composition.all short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.all{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.any short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.any{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.clone short:1 with_parentheses:false{{/auto_link}}            | See {{#auto_link}}pydec.clone{{/auto_link}}                                                                                               |
| {{#auto_link}}pydec.Composition.contiguous short:1 with_parentheses:false{{/auto_link}}       | Returns a contiguous in memory composition containing the same data as `self` composition.                                                |
| {{#auto_link}}pydec.Composition.cpu short:1 with_parentheses:false{{/auto_link}}              | Returns a copy of this object in CPU memory.                                                                                              |
| {{#auto_link}}pydec.Composition.cuda short:1 with_parentheses:false{{/auto_link}}             | Returns a copy of this object in CUDA memory.                                                                                             |
| {{#auto_link}}pydec.Composition.diagonal_scatter short:1 with_parentheses:false{{/auto_link}} | See {{#auto_link}}pydec.diagonal_scatter{{/auto_link}}                                                                                    |
| {{#auto_link}}pydec.Composition.dim short:1 with_parentheses:false{{/auto_link}}              | Returns the number of dimensions of the individual components of `self` composition.                                                      |
| {{#auto_link}}pydec.Composition.div short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.div{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.div_ short:1 with_parentheses:false{{/auto_link}}             | In-place version of {{#auto_link}}pydec.Composition.div short:2{{/auto_link}}                                                             |
| {{#auto_link}}pydec.Composition.gather short:1 with_parentheses:false{{/auto_link}}           | See {{#auto_link}}pydec.gather{{/auto_link}}                                                                                              |
| {{#auto_link}}pydec.Composition.index_fill_ short:1 with_parentheses:false{{/auto_link}}      | Fills the elements of each component of the `self` composition with value `value` by selecting the indices in the order given in `index`. |
| {{#auto_link}}pydec.Composition.index_fill short:1 with_parentheses:false{{/auto_link}}       | Out-of-place version of {{#auto_link}}pydec.Composition.index_fill_ short:2{{/auto_link}}.                                                |
| {{#auto_link}}pydec.Composition.index_select short:1 with_parentheses:false{{/auto_link}}     | See {{#auto_link}}pydec.index_select{{/auto_link}}                                                                                        |
| {{#auto_link}}pydec.Composition.is_contiguous short:1 with_parentheses:false{{/auto_link}}    | Returns True if `self` composition is contiguous in memory in the order specified by memory format.                                       |
| {{#auto_link}}pydec.Composition.masked_scatter_ short:1 with_parentheses:false{{/auto_link}}  | Copies elements from `source` into each component of `self` composition at positions where the `mask` is True.                            |
| {{#auto_link}}pydec.Composition.masked_scatter short:1 with_parentheses:false{{/auto_link}}   | Out-of-place version of {{#auto_link}}pydec.Composition.masked_scatter_ short:2{{/auto_link}}.                                            |
| {{#auto_link}}pydec.Composition.masked_fill_ short:1 with_parentheses:false{{/auto_link}}     | Fills elements of each component of `self` composition with value where `mask` is True.                                                   |
| {{#auto_link}}pydec.Composition.masked_select short:1 with_parentheses:false{{/auto_link}}    | See {{#auto_link}}pydec.masked_select{{/auto_link}}                                                                                       |
| {{#auto_link}}pydec.Composition.mean short:1 with_parentheses:false{{/auto_link}}             | See {{#auto_link}}pydec.mean{{/auto_link}}                                                                                                |
| {{#auto_link}}pydec.Composition.mul short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.mul{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.mul_ short:1 with_parentheses:false{{/auto_link}}             | In-place version of {{#auto_link}}pydec.Composition.mul short:2{{/auto_link}}.                                                            |
| {{#auto_link}}pydec.Composition.numel short:1 with_parentheses:false{{/auto_link}}            | See {{#auto_link}}pydec.numel{{/auto_link}}                                                                                               |
| {{#auto_link}}pydec.Composition.c_numel short:1 with_parentheses:false{{/auto_link}}          | See {{#auto_link}}pydec.c_numel{{/auto_link}}                                                                                             |
| {{#auto_link}}pydec.Composition.numc short:1 with_parentheses:false{{/auto_link}}             | See {{#auto_link}}pydec.numc{{/auto_link}}                                                                                                |
| {{#auto_link}}pydec.Composition.permute short:1 with_parentheses:false{{/auto_link}}          | See {{#auto_link}}pydec.permute{{/auto_link}}                                                                                             |
| {{#auto_link}}pydec.Composition.reshape short:1 with_parentheses:false{{/auto_link}}          | Returns a composition with the same data and number of elements as `self` but with the specified shape of each component.                 |
| {{#auto_link}}pydec.Composition.reshape_as short:1 with_parentheses:false{{/auto_link}}       | Returns this composition as the same shape as `other`.                                                                                    |
| {{#auto_link}}pydec.Composition.round short:1 with_parentheses:false{{/auto_link}}            | See {{#auto_link}}pydec.round{{/auto_link}}                                                                                               |
| {{#auto_link}}pydec.Composition.round_ short:1 with_parentheses:false{{/auto_link}}           | In-place version of {{#auto_link}}pydec.Composition.round short:2{{/auto_link}}.                                                          |
| {{#auto_link}}pydec.Composition.scatter_ short:1 with_parentheses:false{{/auto_link}}         | Writes all values from the tensor `src` into each component of `self` at the indices specified in the `index` tensor.                     |
| {{#auto_link}}pydec.Composition.scatter short:1 with_parentheses:false{{/auto_link}}          | Out-of-place version of {{#auto_link}}pydec.Composition.scatter_ short:2{{/auto_link}} .                                                  |
| {{#auto_link}}pydec.Composition.select short:1 with_parentheses:false{{/auto_link}}           | See {{#auto_link}}pydec.select{{/auto_link}}                                                                                              |
| {{#auto_link}}pydec.Composition.size short:1 with_parentheses:false{{/auto_link}}             | Returns the size of the individual components of the `self` composition.                                                                  |
| {{#auto_link}}pydec.Composition.squeeze short:1 with_parentheses:false{{/auto_link}}          | See {{#auto_link}}pydec.squeeze{{/auto_link}}                                                                                             |
| {{#auto_link}}pydec.Composition.squeeze_ short:1 with_parentheses:false{{/auto_link}}         | In-place version of {{#auto_link}}pydec.Composition.squeeze short:2{{/auto_link}} .                                                       |
| {{#auto_link}}pydec.Composition.sub short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.sub{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.sub_ short:1 with_parentheses:false{{/auto_link}}             | In-place version of {{#auto_link}}pydec.Composition.sub short:2{{/auto_link}}.                                                            |
| {{#auto_link}}pydec.Composition.sum short:1 with_parentheses:false{{/auto_link}}              | See {{#auto_link}}pydec.sum{{/auto_link}}                                                                                                 |
| {{#auto_link}}pydec.Composition.c_sum short:1 with_parentheses:false{{/auto_link}}            | See {{#auto_link}}pydec.c_sum{{/auto_link}}                                                                                               |
| {{#auto_link}}pydec.Composition.to short:1 with_parentheses:false{{/auto_link}}               | Performs Tensor dtype and/or device conversion.                                                                                           |
| {{#auto_link}}pydec.Composition.transpose short:1 with_parentheses:false{{/auto_link}}        | See {{#auto_link}}pydec.transpose{{/auto_link}}                                                                                           |
| {{#auto_link}}pydec.Composition.transpose_ short:1 with_parentheses:false{{/auto_link}}       | In-place version of {{#auto_link}}pydec.Composition.transpose short:2{{/auto_link}}.                                                      |
| {{#auto_link}}pydec.Composition.type short:1 with_parentheses:false{{/auto_link}}             | Returns the type if *dtype* is not provided, else casts this object to the specified type.                                                |
| {{#auto_link}}pydec.Composition.type_as short:1 with_parentheses:false{{/auto_link}}          | Returns this composition cast to the type of the given tensor or composition.                                                             |
| {{#auto_link}}pydec.Composition.unsqueeze short:1 with_parentheses:false{{/auto_link}}        | See {{#auto_link}}pydec.unsqueeze{{/auto_link}}                                                                                           |
| {{#auto_link}}pydec.Composition.unsqueeze_ short:1 with_parentheses:false{{/auto_link}}       | In-place version of {{#auto_link}}pydec.Composition.unsqueeze short:2{{/auto_link}}.                                                      |
| {{#auto_link}}pydec.Composition.view short:1 with_parentheses:false{{/auto_link}}             | Returns a new composition with the same data as the `self` composition but of a different `shape`.                                        |
| {{#auto_link}}pydec.Composition.view_as short:1 with_parentheses:false{{/auto_link}}          | View this composition as the same shape as `other`.                                                                                       |
