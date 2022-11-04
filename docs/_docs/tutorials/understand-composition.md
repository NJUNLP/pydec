---
title: "Tutorials: Understand Composition"
tags: 
 - tutorial
 - composition
description: Getting started with PyDec
---
# Tutorials: Understand Composition

## Decomposition

Given an arbitrary tensor $h$ in the network, there exists a way to decompose it into the sum of several components. As shown in the following equation,

$$
h = \frac{\mathscr{D}h}{\mathscr{D}x_1} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}x_m},
$$

where $x_1,\cdots,x_m$ are the inputs to the network. Each component corresponds to one or some of the inputs (marked in the denominator), e.g., the components of $h$ corresponding to $\{x_1, x_2\}$ are denoted as $\frac{\mathscr{D}h}{\mathscr{D}x_1\mathscr{D}x_2}$

## Composition
In PyDec, we use the data structure (i.e., Composition) to store the components, i.e., the right part of the above equation, while the left part can be obtained simply by summing the components.

## Create a Composition
```python
>>> import pydec
```

**From size and component number**
```python
>>> size = (3, 2)
>>> component_num = 4
>>> c = pydec.Composition(size, component_num)
```
This creates a composition containing 4 tensors of size (3, 2), initialized with 0.

**From another Composition**
```python
>>> c_copy = pydec.Composition(c)
```
This will clone an identical c, but without preserving any computational graph.

**From component tensors**
```python
>>> component_num = 4
>>> c_size = (component_num, 3, 2)
>>> t = torch.randn(c_size) # 4 x 3 x 2
>>> c_tensor = pydec.Composition(t)
```
This also creates a composition containing 4 tensors of size (3, 2), initialized with tensor t.

## Initialize a Composition
After creating a Composition, we usually initialize the value of the Composition based on orthogonality, i.e.,

$$
\frac{\mathscr{D}x_i}{\mathscr{D}x_j}=\begin{cases}x_i, &\text{if}\ i=j\newline 0, &\text{otherwise}\end{cases}.
$$

**By assign**

```python
# The input consists of x0 and x1, each containing a tensor with a feature number of 2.
>>> size = (2,)
>>> x0 = torch.randn(size)
>>> x1 = torch.randn(size)
>>> c0 = pydec.Composition(size, component_num=2)
>>> c1 = pydec.Composition(size, component_num=2)
# Initialize by assign
>>> c0[0] = x0
>>> c1[1] = x1
```
*You can access the components by indices, e.g. `c[0]` and `c[3:5]`.*

**By diagonal scatter**

In practice, usually all inputs are batched into a tensor. Therefore a more useful initialization method is based on the `torch.diagonal_scatter` function.
```python
>>> component_num = 3
>>> size = (3, 2)
>>> x = torch.randn(size)
tensor([[-0.4682,  1.2375],
        [ 0.7185,  0.2311],
        [-0.4043, -1.5946]])
>>> c = pydec.Composition(size, component_num)
>>> c = pydec.diagonal_init(c, src=x, dim=0)
composition 0:
tensor([[-0.4682,  1.2375],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000]])
composition 1:
tensor([[0.0000, 0.0000],
        [0.7185, 0.2311],
        [0.0000, 0.0000]])
composition 2:
tensor([[ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [-0.4043, -1.5946]])
residual:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
```

## Attributes of a Composition
**Size**

`Composition.size()` returns the shape of each composition tensor, `Composition.c_size()` returns the shape whose first dimension is the number of components.
```python
>>> c = pydec.Composition((3, 2), component_num=4)
>>> c.size()
torch.Size([3, 2])
>>> c.c_size()
torch.Size([4, 3, 2])
```

`len()` and `Composition.numc()` return the number of components.
```python
>>> len(c)
4
>>> c.numc()
4
```

## Residual of a Composition

The decomposition of the tensor in the network with respect to the inputs is usually not complete, resulting in a residual component which represents the contribution of the bias parameters of the model (usually used in the linear layer), i.e.,

$$
h = \frac{\mathscr{D}h}{\mathscr{D}x_1} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}x_m} + \frac{\mathscr{D}h}{\mathscr{D}b^1\cdots\mathscr{D}b^L},
$$

where $b$ is the bias parameters of the model. 

By default, the term $\frac{\mathscr{D}h}{\mathscr{D}b^1\cdots\mathscr{D}b^L}$ is saved to `Composition._residual`. PyDec configures some strategies to reallocate the bias term to the components corresponding to the input (see {% include doc.html name="Bias Decomposition" path="notes/bias-decomposition" %}, so the residual may be 0 or some other value (depending on the reallocation strategy).


## Operations on Compositions

We have implemented some common operations on Compositions, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing).

Most of the operations are the same as tensor operations, and a convenient expression to understand is that performing one operation on a composition is equivalent to performing the same operation for all components of the composition, including the residual component. More details about the operations can be found {% include doc.html name="here" path="notes/overloaded-operators" %}.

Example:
```python
>>> c
composition 0:
tensor([[1., 1., 1., 1.],
        [0., 0., 0., 0.]])
composition 1:
tensor([[0., 0., 0., 0.],
        [1., 1., 1., 1.]])
residual:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.]])
>>> 3 * c # multiply
composition 0:
tensor([[3., 3., 3., 3.],
        [0., 0., 0., 0.]])
composition 1:
tensor([[0., 0., 0., 0.],
        [3., 3., 3., 3.]])
residual:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.]])
>>> c + c # add
composition 0:
tensor([[2., 2., 2., 2.],
        [0., 0., 0., 0.]])
composition 1:
tensor([[0., 0., 0., 0.],
        [2., 2., 2., 2.]])
residual:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.]])
>>> W = torch.randn((4,3))
>>> W
tensor([[-0.4682,  1.2375,  0.7185],
        [ 0.2311, -0.4043, -1.5946],
        [-0.4981,  0.2654,  0.0849],
        [ 1.0203, -0.4293, -0.2616]])
>>> c @ W # matmul
composition 0:
tensor([[ 0.2851,  0.6694, -1.0529],
        [ 0.0000,  0.0000,  0.0000]])
composition 1:
tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.2851,  0.6694, -1.0529]])
residual:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
