---
title: "pydec"
description: API for the module pydec
---

# PYDEC

The pydec package contains data structures for compositions and defines mathematical operations over these compositions. Additionally, it provides many useful utilities. Our documentation structure is referenced from Pytorch.

## Compositions

| API                                                                       | Description                                                                                   |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| {% include codelink.html name="numel" path="pythonapi/pydec/numel" %}     | Returns the total number of elements in the individual components of the `input` composition. |
| {% include codelink.html name="c_numel" path="pythonapi/pydec/c_numel" %} | Returns the total number of elements in all components of the `input` composition.            |
| {% include codelink.html name="numc" path="pythonapi/pydec/numc" %}       | Returns the number of components in the `input` composition.                                  |

## Creation Ops
To create Composition by class, use {% include codelink.html name="pydec.Composition" path="pythonapi/pydec.Composition" %}.

| API                                                                             | Description                                                                                             |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| {% include codelink.html name="void" path="pythonapi/pydec/void" %}             | Returns an empty composition.                                                                           |
| {% include codelink.html name="zeros_like" path="pythonapi/pydec/zeros_like" %} | Returns a composition whose components filled with the scalar value *0*, with the same size as `input`. |

## Indexing, Slicing, Joining, Mutating Ops

| API                                                                                         | Description                                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| {% include codelink.html name="cat" path="pythonapi/pydec/cat" %}                           | Concatenates every component of the given sequence of `seq` compositions in the given dimension.                                                             |
| {% include codelink.html name="concat" path="pythonapi/pydec/concat" %}                     | Alias of {% include codelink.html name="cat" path="pythonapi/pydec/cat" %}.                                                                                  |
| {% include codelink.html name="concatenate" path="pythonapi/pydec/concatenate" %}           | Alias of {% include codelink.html name="cat" path="pythonapi/pydec/cat" %}.                                                                                  |
| {% include codelink.html name="c_cat" path="pythonapi/pydec/c_cat" %}                       | Concatenates the given sequence of `seq` compositions in the component dimension.                                                                            |
| {% include codelink.html name="gather" path="pythonapi/pydec/gather" %}                     | Gathers values along an axis specified by *dim*.                                                                                                             |
| {% include codelink.html name="index_select" path="pythonapi/pydec/index_select" %}         | Returns a new composition which indexes the `input` tensor along dimension `dim` using the entries in `index` which is a LongTensor.                         |
| {% include codelink.html name="c_index_select" path="pythonapi/pydec/c_index_select" %}     | Returns a new composition which indexes the `input` tensor along the component dimension using the entries in `index` which is a LongTensor.                 |
| {% include codelink.html name="masked_select" path="pythonapi/pydec/masked_select" %}       | Returns a new composition with 1-D component tensor which indexes the `input` composition according to the boolean mask `mask` which is a BoolTensor.        |
| {% include codelink.html name="permute" path="pythonapi/pydec/permute" %}                   | Returns a view of the original composition `input` with its components' dimensions permuted.                                                                 |
| {% include codelink.html name="reshape" path="pythonapi/pydec/reshape" %}                   | Returns a composition with the same data and number of elements as `input`, but with the specified shape.                                                    |
| {% include codelink.html name="diagonal_scatter" path="pythonapi/pydec/diagonal_scatter" %} | Embeds the values of the `src` tensor into `input` composition along the diagonal elements of every component of `input`, with respect to `dim1` and `dim2`. |
| {% include codelink.html name="diagonal_init" path="pythonapi/pydec/diagonal_init" %}       | Embeds the values of the `src` tensor into `input` composition along the diagonal components of `input`, with respect to `dim`.                              |
| {% include codelink.html name="squeeze" path="pythonapi/pydec/squeeze" %}                   | Returns a composition with all the dimensions of components of `input` of size *1* removed.                                                                  |
| {% include codelink.html name="stack" path="pythonapi/pydec/stack" %}                       | Concatenates every component of a sequence of compositions along a new dimension.                                                                            |
| {% include codelink.html name="transpose" path="pythonapi/pydec/transpose" %}               | Returns a composition that is a transposed version of `input`.                                                                                               |
| {% include codelink.html name="unsqueeze" path="pythonapi/pydec/unsqueeze" %}               | Returns a new composition with a dimension of size one inserted at the specified position of each component.                                                 |




