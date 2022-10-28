---
layout: page
title:  "Tracking Forward Propagation"
category: notes
date: 2022-10-29 1:00:00
order: 2
---

To obtain the decomposition of the output or intermediate variables, the input to the network is first wrapped with Composition, and then the linear transformations and operations defined by the network are applied to Composition. Composition automatically maintains the decomposition of the corresponding variables of the original network.

## Initializing Compositions
Create a corresponding Composition for each tensor that is fed into the network instead of for each input variable, e.g. if all inputs are organized in a certain dimension in a tensor, then only a Composition needs to be created for this tensor.

The number of components in Composition is determined by the user's needs, such as creating a Composition with 2 components corresponding to the first 10 tokens and the remaining tokens of the sentence, or setting the number of components to the number of features to observe the impact of different features. For attribution analysis, a more common setup is to set the number of components to the number of tokens of text or the number of pixels of an image.

Suppose the input is an embedding representation of a piece of text, with a batch size of 16, a text length of 20, and 512 embedding features:

```python
>>> input = torch.randn((16, 20, 512))
```

Initializing to decompose in the sentence length dimension：
```python
>>> c = pydec.Composition((16, 20, 512), component_num=20)
>>> c = pydec.diagonal_init(c, src=input, dim=1)
```

Initializing to decompose in the feature dimension:
```python
>>> c = pydec.Composition((16, 20, 512), component_num=512)
>>> c = pydec.diagonal_init(c, src=input, dim=2)
```

Initializing to decompose in the joint feature dimension and sentence length dimension:
```python
>>> c = pydec.Composition((16, 20, 512), component_num=20*512)
>>> c = c.view(16, 20*512)
>>> c = pydec.diagonal_init(c, src=input.view(16,20*512), dim=1)
>>> c = c.view_as(input)
```

If you want to compute the decomposition in training and keep the computational graph of the components. Do not use the `requires_grad` parameter in the constructor of Composition, otherwise the initialization of Composition as a leaf node cannot be completed by assignment. It is recommended to assign the input with gradient to the Composition without gradient.

## Forward Compositions

Use the operations provided by PyDec to complete the forward computation. `pydec.nn` also provides some wrapped high-level components. Since the decomposition is usually done in the Inference phase, it is recommended to use the functions provided by `pydec.nn.functional`.

To get a decomposition of the output or intermediate variables, mimic the operation you performed on the input in the forward function on the initialized Composition.

For example, you get the tensor `h3` by the following operation:

```python
h3 = -h1 @ W + 3 * h2.permute(-1,0,1)
```

Then you need to perform the following operation on the Composition corresponding to `h1` and `h2` to get the Composition of `h3`.

```python
c3 = -c1 @ W + 3 * c2.permute(-1,0,1)
```

We have implemented a number of common functions for `pydec.Composition`. In most cases you just need to use the same functions and parameters to complete the trace. In most cases you just need to use the same functions and arguments to complete the trace. Other functions related to component operations start with `c_`. If you use a function not yet provided by PyDec, you may need to use other functions to do the equivalent operation or implement the function yourself (PR is welcome).
