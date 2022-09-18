[![Test](https://github.com/DoubleVII/pydec/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/DoubleVII/pydec/actions/workflows/python-package-conda.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center" style="font-size:60px">
  PyDec
</h1>

PyDec is a linear decomposition toolkit for neural network based on [PyTorch](https://pytorch.org/), which can decompose the tensor in the forward process into given components with a small amount of code. The result of decomposition can be applied to tasks such as attribution analysis.

# Requirements and Installation
* [PyTorch](https://pytorch.org/) version >= 1.11.0
* Python version >= 3.6
* To install PyDec and develop locally:

``` bash
git clone https://github.com/DoubleVII/pydec
cd pydec
pip install --editable ./
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
h = \frac{\mathscr{D}h}{\mathscr{D}x_1} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}x_m} + \frac{\mathscr{D}h}{\mathscr{D}b^1\cdots\mathscr{D}b^L},
$$

where $x_1,\cdots,x_m$ are the inputs to the network and $b$ is the bias parameter of the network (usually used in the linear layer). Each component corresponds to one or some of the inputs (marked in the denominator), e.g., the components of $h$ corresponding to $\{x_1, x_2\}$ are denoted as $\frac{\mathscr{D}h}{\mathscr{D}x_1\mathscr{D}x_2}$

## Composition
In PyDec, we use the data structure (i.e., Composition) to store the components, i.e., the right part of the above equation, while the left part can be obtained simply by summing the components.

### Create a Composition
```python
import pydec
```

**From size and component number**
```python
size = (3, 2)
component_num = 4
c = pydec.Composition(size, component_num)
```
This creates a composition containing 4 tensors of size (3, 2), initialized with 0.

**From another Composition**
```python
c_copy = pydec.Composition(c)
```
This will clone an identical c, but without preserving any computational graph.

**From component tensors**
```python
component_num = 4
c_size = (component_num, 3, 2)
t = torch.randn(c_size) # 4 x 3 x 2

c_tensor = pydec.Composition(t)
```
This also creates a composition containing 4 tensors of size (3, 2), initialized with tensor t.

### Initialize a Composition
After creating a Composition, we usually initialize the value of the Composition based on orthogonality, i.e.,
$$
\frac{\mathscr{D}x_i}{\mathscr{D}x_j}=\begin{cases}x_i, &\text{if}\ i=j\\\boldsymbol{0}, &\text{otherwise}\end{cases}.
$$

**By assign**
```python
size = (3, 2)
x0 = torch.randn(size)
x1 = torch.randn(size)
c = pydec.Composition(size, component_num=2)

c[0] = x0
c[1] = x2
```
*You can access the components by indices, e.g. `c[0]` and `c[3:5]`.*

**By diagonal scatter**
In practice, usually all inputs are batched into a tensor. Therefore a more useful initialization method is based on the `torch.diagonal_scatter` function.
```python
component_num = 3
size = (3, 2)
x = torch.randn(size)
c = pydec.Composition((3, 2), component_num)
c = pydec.diagonal_init(c, src=x, dim=0)
```
Out:
```python
>>> x
    tensor([[-0.4682,  1.2375],
           [ 0.7185,  0.2311],
           [-0.4043, -1.5946]])
>>> composition 0:
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

## Operations on Compositions
We have implemented some common operations on Compositions, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing).

Most of the operations are the same as tensor operations, and a convenient expression to understand is that performing one operation on a composition is equivalent to performing the same operation for all components of the composition.


# Documentation

We will release latter.