![PyPI](https://img.shields.io/pypi/v/pydec)
[![Test](https://github.com/DoubleVII/pydec/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/DoubleVII/pydec/actions/workflows/python-package-conda.yml)
![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DoubleVII/pydec/python-coverage-comment-action-data/endpoint.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



<h1 align="center" style="font-size:60px">
  PyDec
</h1>

PyDec is a linear decomposition toolkit for neural network based on [PyTorch](https://pytorch.org/), which can decompose the tensor in the forward process into given components with a small amount of code. The result of decomposition can be applied to tasks such as attribution analysis.

# Examples
## Attribution
Contribution Heat maps of the Roberta model (fine-tuned on SST-2). Warm colors indicate high
contribution while cool colors indicate low contribution. The outputs of the model were positive, negative and positive, but the latter two samples did not match the labels.

<div align="center">
<img src="./docs/source/_static/img/pydec_demo1.png" width="70%">
</div>

## Data flow visualization

![demo2](./docs/source/_static/img/pydec_demo2_1.gif)

![demo2](./docs/source/_static/img/pydec_demo2_2.gif)

# Requirements and Installation
* [PyTorch](https://pytorch.org/) version >= 1.11.0
* Python version >= 3.7
* To install PyDec and develop locally:

``` bash
git clone https://github.com/DoubleVII/pydec
cd pydec
pip install --editable ./
```

* To install the latest stable release:
``` bash
pip install pydec
```

# Getting Started

## Example: deompose a tiny network

Suppose a simple feedforward neural network containing two input tensors and outputting one tensor.
```python
class NN(nn.Module):
    def __init__(self) -> None:...

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        x1 = self.linear1(x1)
        x1 = self.relu(x1)

        x2 = self.linear2(x2)
        x2 = self.relu(x2)

        out = self.linear3(x1+x2)
        return out
```
In order to keep track of the components of inputs x1 and x2 in each hidden tensor, simply initialize the corresponding compositions and apply the same operation for them.
```python
class NN(nn.Module):
    def __init__(self) -> None:...

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.linear1(x1)
        x1 = self.relu(x1)

        x2 = self.linear2(x2)
        x2 = self.relu(x2)

        out = self.linear3(x1 + x2)

        import pydec
        from pydec import Composition
        # Initialize composition
        c1 = Composition(x1.size(), component_num=2).to(x1)
        c1[0] = x1 # Assign x1 to the first component of c1.

        c2 = Composition(x2.size(), component_num=2).to(x2)
        c2[1] = x2 # Assign x2 to the second component of c2.

        # Apply the same operation for composition
        c1 = pydec.nn.functional.linear(
            c1, weight=self.linear1.weight, bias=self.linear1.bias
        )
        c1 = pydec.nn.functional.relu(c1)

        c2 = pydec.nn.functional.linear(
            c2, weight=self.linear2.weight, bias=self.linear2.bias
        )
        c2 = pydec.nn.functional.relu(c2)
        
        c_out = pydec.nn.functional.linear3(
            c1 + c2, weight=self.linear3.weight, bias=self.linear3.bias
        )
        return out, c_out
```

In the above example, each composition consists of two components whose sum is always equal to the corresponding tensor being decomposed, e.g., $x_1=c_1[0]+c_1[1]$ and $out=c_{out}[0]+c_{out}[1]$. Usually, you can think of $c_{out}[i]$ as the contribution of $x_i$ to the tensor $out$.

# Tutorials: understand composition

## Decomposition

Given an arbitrary tensor $h$ in the network, there exists a way to decompose it into the sum of several components. As shown in the following equation,

$$
h = \frac{\mathscr{D}h}{\mathscr{D}x_1} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}x_m},
$$

where $x_1,\cdots,x_m$ are the inputs to the network. Each component corresponds to one or some of the inputs (marked in the denominator), e.g., the components of $h$ corresponding to $\{x_1, x_2\}$ are denoted as $\frac{\mathscr{D}h}{\mathscr{D}x_1\mathscr{D}x_2}$

## Composition
In PyDec, we use the data structure (i.e., Composition) to store the components, i.e., the right part of the above equation, while the left part can be obtained simply by summing the components.

### Create a Composition
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

### Initialize a Composition
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
>>> c = pydec.Composition(size, component_num)
>>> c = pydec.diagonal_init(c, src=x, dim=0)
```
Out:
```python
>>> x
    tensor([[-0.4682,  1.2375],
           [ 0.7185,  0.2311],
           [-0.4043, -1.5946]])
>>> c
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
>>> c.size()
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

By default, the term $\frac{\mathscr{D}h}{\mathscr{D}b^1\cdots\mathscr{D}b^L}$ is saved to `Composition._residual`. PyDec configures some strategies to reallocate the bias term to the components corresponding to the input (see [Bias decomposition](#bias-decomposition)), so the residual may be 0 or some other value (depending on the reallocation strategy).


## Operations on Compositions

We have implemented some common operations on Compositions, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing).

Most of the operations are the same as tensor operations, and a convenient expression to understand is that performing one operation on a composition is equivalent to performing the same operation for all components of the composition, including the residual component. More details about the operations can be found here (TODO).

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

## Bias decomposition

In order to reallocate bias term, PyDec will assign them to components other than residual whenever it encounters bias addition operation.

Assume that $h^\prime=h+b$ and $h$ is denoted as the sum of $m$ components, i.e., $h=c_1+\cdots+c_m$. Then $b$ is decomposed into $m$ parts and added to each of the $m$ components:

$$
b=p_1+\cdots+p_m \newline
c^\prime_i=c_i+p_i.
$$
The decomposition of $h^\prime$ was thus obtained as $h^\prime=c^\prime_1+\cdots+c^\prime_m$

PyDec has some built-in strategies to decompose bias, and they mostly calculate $p_i$ based on the value of $c_i$. By default, PyDec just adds bias to residual component without performing any bias decomposition. More details about the bias decomposition can be found here (TODO).

## Tracking forward propagation

To obtain the decomposition of the output or intermediate variables, the input to the network is first wrapped with Composition, and then the linear transformations and operations defined by the network are applied to Composition. Composition automatically maintains the decomposition of the corresponding variables of the original network.

### Initializing Compositions
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

### Forward Compositions

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

## Error control

Although in theory the recovery from Composition is exactly equivalent to the ground truth. However, in practice, there will be errors brought by the computer. Especially in deep networks, the error may be magnified to an unacceptable degree. We give some suggestions for reducing errors and provide tools for error checking.

### Error reduction

Our most recommended method for reducing errors is to use double precision computations, usually by simply adding `model=model.double()` after the model is loaded. If you enable double precision calculations, the error from the decomposition is almost negligible.

When double precision computation cannot be enabled for speed and memory reasons, you may consider adding the error term to Composition as bias. Depending on the bias reallocation policy, the error is added to the residual or assigned to each component.

You can even consider making the ground truth equal to Composition's recovery, but this may change the classification result of the network.

### Error checking
You can use `PyDec.check_error` to check the error of the given Composition and reference. In order to provide ground truth as a reference, you usually need to keep the forward process of the original network. We recommend that you use it often during development, not only for error control, but also to help you find bugs in your code.

# Documentation

We will release latter.