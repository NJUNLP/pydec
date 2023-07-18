# Tutorials: Understand Composition

## Decomposition

Given an arbitrary tensor $h$ in the network, there exists a way to decompose it into the sum of several components. As shown in the following equation,

$$
h = \frac{\mathscr{D}h}{\mathscr{D}\{x_1\}} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}\{x_m\}},
$$

where $x_1,\cdots,x_m$ are the inputs to the network. Each component corresponds to one or some of the inputs (marked in the denominator), e.g., the components of $h$ corresponding to $\{x_1, x_2\}$ are denoted as $\frac{\mathscr{D}h}{\mathscr{D}\{x_1,x_2\}}$

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
>>> c = pydec.zeros(size, component_num)
```
This creates a composition containing 4 tensors of size (3, 2), initialized with 0.

**From another Composition**
```python
>>> c_copy = pydec.Composition(c)
```
This will clone an identical `c`, but without preserving any computational graph.

**From component tensors**
```python
>>> component_num = 4
>>> c_size = (component_num, 3, 2)
>>> t = torch.randn(c_size) # 4 x 3 x 2
>>> c_tensor = pydec.Composition(t)
>>> c_tensor.c_size()
"""
torch.Size([4, 3, 2])
"""
>>> c_tensor.size()
"""
torch.Size([3, 2])
"""
```
This also creates a composition containing 4 tensors of size (3, 2), initialized with tensor t.

## Initializing a Composition
After creating a Composition, we usually initialize the value of the Composition based on orthogonality, i.e.,

$$
\frac{\mathscr{D}x_i}{\mathscr{D}g}=\begin{cases}x_i, &\text{if}\ x_i\in g\\\boldsymbol{0}, &\text{otherwise}\end{cases}.
$$

**By assign**

```python
>>> size = (2, 32)
>>> X = torch.randn(size)
# The input consists of x0 and x1, each containing a tensor with a feature number of 32.
>>> x0, x1 = X[0], X[1]
>>> C = pydec.zeros(size, c_num=2)
# Initialize by assign
>>> with pydec.enable_c_accessing():
...     C[0, 0] = x0
...     C[1, 1] = x1
...     print("component 0:\n", C[0]) 
...     print("component 1:\n", C[1])
"""
component 0:
 tensor([[ 0.1711, -0.8391, -0.7864, -0.1591,  0.1729,  0.1396,  0.1319, -0.6490,
         -1.9130,  2.9762, -0.8694, -0.0117, -1.4587, -0.0112, -1.1870, -2.3948,
         -1.5261,  1.3400, -2.1850, -0.8882,  0.1124, -0.1577, -0.3339, -0.5000,
          0.1017,  0.4712,  0.7827, -1.2579, -0.2937,  1.7840,  1.6399,  0.0031],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
component 1:
 tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.1231, -1.0988, -0.4779,  2.2519,  1.3462, -0.8508,  0.5292,  0.6132,
          0.1049,  0.5999, -0.4996, -0.3270,  0.0688, -0.4993,  0.9848, -0.2268,
          1.0324, -1.4307, -1.3290, -1.8317, -1.4679,  1.4789, -1.0039, -0.0492,
          0.0694,  0.4789, -0.2259,  0.2443, -0.6499, -0.8895,  0.5010, -0.2561]])
"""
```
?> For component accessing, refer to [Component Accessing](#component-accessing)

<!-- You can access the components by indices in the `enable_c_accessing` context, e.g. `C[0]` and `C[3:5]`.  -->

**By diagonal scatter**

In practice, usually all inputs are batched into a tensor. Therefore a more useful initialization method is based on the {{#auto_link}}pydec.diagonal_init{{/auto_link}} function.
```python
>>> component_num = 3
>>> size = (3, 2)
>>> x = torch.randn(size)
>>> x
"""
tensor([[-0.7527,  0.8432],
        [-2.0405, -0.7070],
        [-0.7259, -0.3996]])
"""
>>> c = pydec.zeros(size, component_num)
>>> c = pydec.diagonal_init(c, src=x, dim=0)
>>> c
"""
composition{
  components:
    tensor([[-0.7527,  0.8432],
            [ 0.0000,  0.0000],
            [ 0.0000,  0.0000]]),
    tensor([[ 0.0000,  0.0000],
            [-2.0405, -0.7070],
            [ 0.0000,  0.0000]]),
    tensor([[ 0.0000,  0.0000],
            [ 0.0000,  0.0000],
            [-0.7259, -0.3996]]),
  residual:
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])}
"""
```
## Component accessing
Under normal circumstances, operations on compositions will only be carried out on the tensor dimensions, so as not to interfere with its transformation in the model. If you need to access its components, pydec provides context managers {{#auto_link}}pydec.enable_c_accessing{{/auto_link}} and {{#auto_link}}pydec.set_c_accessing_enabled{{/auto_link}} to achieve this.

In the context of component accessing, the first dimension of indices is used to access components. If indexing a single component, a tensor is returned. If indexing multiple components, they are returned as a composition. Some functions, such as `len()` and `iter()`, are also affected.

```python
>>> c = pydec.Composition(torch.rand((3,4)))
>>> c
"""
composition{
  components:
    tensor([0.5818, 0.3793, 0.1772, 0.0045]),
    tensor([0.1867, 0.1313, 0.3425, 0.1393]),
    tensor([0.1749, 0.8727, 0.2270, 0.1390]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> c[0]
"""
composition{
  components:
    tensor(0.5818),
    tensor(0.1867),
    tensor(0.1749),
  residual:
    tensor(0.)}
"""
>>> c[0:2]
"""
composition{
  components:
    tensor([0.5818, 0.3793]),
    tensor([0.1867, 0.1313]),
    tensor([0.1749, 0.8727]),
  residual:
    tensor([0., 0.])}
"""
>>> len(c)
"""
4
"""
>>> with pydec.enable_c_accessing():
...     c[0]
...     c[0:1]
...     c[0:2, :2]
...     len(c)
"""
tensor([0.5818, 0.3793, 0.1772, 0.0045])
composition{
  components:
    tensor([0.5818, 0.3793, 0.1772, 0.0045]),
  residual:
    tensor([0., 0., 0., 0.])}
composition{
  components:
    tensor([0.5818, 0.3793]),
    tensor([0.1867, 0.1313]),
  residual:
    tensor([0., 0.])}
3
"""
```
For convenience, pydec also provides a syntax sugar that eliminates the need for context managers. You only need to add a pair of parentheses after the composition being accessed:
```python
>>> c()[0] = -1
>>> c()[:-1]
"""
composition{
  components:
    tensor([-1., -1., -1., -1.]),
    tensor([0.4141, 0.7358, 0.7060, 0.7372]),
  residual:
    tensor([0., 0., 0., 0.])}
"""
>>> len(c())
"""
3
"""
```

## Attributes of a Composition
**Size**

{{#auto_link}}pydec.Composition.size short:1{{/auto_link}} returns the shape of each composition tensor; {{#auto_link}}pydec.Composition.c_size short:1{{/auto_link}} returns the shape whose first dimension is the number of components.
```python
>>> c = pydec.zeros((3, 2), c_num=4)
>>> c.size()
"""
torch.Size([3, 2])
"""
>>> c.c_size()
"""
torch.Size([4, 3, 2])
"""
```

`len()` in the *enable_c_accessing* context and {{#auto_link}}pydec.Composition.numc short:1{{/auto_link}} return the number of components.
```python
>>> len(c())
"""
4
"""
>>> c.numc()
"""
4
"""
```

## Residual of a Composition

The decomposition of the tensor in the network with respect to the inputs is usually not complete, resulting in a residual component which represents the contribution of the bias parameters of the model (usually used in the linear layer), i.e.,

$$
h = \frac{\mathscr{D}h}{\mathscr{D}\{x_1\}} + \cdots + \frac{\mathscr{D}h}{\mathscr{D}\{x_m\}} + \frac{\mathscr{D}h}{\mathscr{D}\{b^1\cdots\mathscr{D}b^L\}},
$$

where $b$ is the bias parameters of the model. 

This term $\frac{\mathscr{D}h}{\mathscr{D}\{b^1\cdots\mathscr{D}b^L\}}$ is saved to {{#auto_link}}pydec.Composition.residual short:1 with_parentheses:false{{/auto_link}}.


## Operations on Compositions

We have implemented some common operations on Compositions, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing).

Most of the operations are the same as tensor operations, and a rough way to understand is that performing one operation on a composition is equivalent to performing the same operation for all components of the composition, including the residual component.

Example:
```python
>>> c
"""
composition{
  components:
    tensor([[1., 1., 1., 1.],
            [0., 0., 0., 0.]]),
    tensor([[0., 0., 0., 0.],
            [1., 1., 1., 1.]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> 3 * c # multiply
"""
composition{
  components:
    tensor([[3., 3., 3., 3.],
            [0., 0., 0., 0.]]),
    tensor([[0., 0., 0., 0.],
            [3., 3., 3., 3.]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> c + c # add
"""
composition{
  components:
    tensor([[2., 2., 2., 2.],
            [0., 0., 0., 0.]]),
    tensor([[0., 0., 0., 0.],
            [2., 2., 2., 2.]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> W = torch.randn((4,3))
>>> W
"""
tensor([[-1.7442,  0.6521,  0.8905],
        [ 0.1515,  1.7411, -0.3745],
        [-1.5157,  0.0436,  1.1825],
        [ 1.1856, -1.2636, -1.7592]])
"""
>>> c @ W # matmul
"""
composition{
  components:
    tensor([[-1.9229,  1.1732, -0.0607],
            [ 0.0000,  0.0000,  0.0000]]),
    tensor([[ 0.0000,  0.0000,  0.0000],
            [-1.9229,  1.1732, -0.0607]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
```
