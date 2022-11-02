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

{% include alert.html type="NOTE" content="To create Composition by class, please refer to {% include codelink.html name="pydec.Composition" path="pythonapi/pydec.Composition#composition-class-reference" %}" %}


<!-- There is no creation ops implemented yet, to create compositions, use . -->

| API                                                                             | Description                                                                          |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| {% include codelink.html name="void" path="pythonapi/pydec/void" %}             | Returns an empty composition.                                                        |
| {% include codelink.html name="zeros_like" path="pythonapi/pydec/zeros_like" %} | Returns a composition filled with the scalar value 0, with the same size as `input`. |