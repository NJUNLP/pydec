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
<img src="./docs/assets/img/pydec_demo1.png" width="70%">
</div>

## Data flow visualization

![demo2](./docs/assets/img/pydec_demo2_1.gif)

![demo2](./docs/assets/img/pydec_demo2_2.gif)

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

# Documentation

The [full documentation](https://doublevii.github.io/pydec/) contains examples of implementations on realistic models, tutorials, notes and Python API.