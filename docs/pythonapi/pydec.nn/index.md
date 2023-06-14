# PYDEC.NN

## Basic building blocks
PyDec does not have any built-in network modules, but is compatible with modules in PyTorch. For example, you can input a composition to a linear layer instance, just like input a tensor:
```python
>>> linear = torch.nn.Linear(3, 3)
>>> input = torch.randn(3)
>>> c_input = pydec.zeros(3, c_num=3)
>>> c_input = pydec.diagonal_init(c_input, input, dim=0)
>>> linear(c_input)
"""
composition{
  components:
    tensor([-0.2307,  0.2061, -0.0232]),
    tensor([-0.0238,  0.1634, -0.0152]),
    tensor([ 0.5997,  0.6379, -0.4860]),
  residual:
    tensor([-0.2243, -0.2790,  0.3018]),
  grad_fn=<AddBackward0>}
"""
```

### The compatible network modules
#### Containers
* [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
* [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential)
* [ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList)
* [ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict)

#### Convolution Layers
* [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
* [nn.LazyConv2d](https://pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d)

#### Pooling layers
* [nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)

#### Non-linear Activations (weighted sum, nonlinearity)
* [nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU)
* [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)
* [nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU)
* [nn.Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)
* [nn.Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)

#### Non-linear Activations (other)
* [nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax)

#### Normalization Layers
* [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)

#### Recurrent Layers
* [nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN)

#### Linear Layers
* [nn.Identity](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html#torch.nn.Identity)
* [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
* [nn.LazyLinear](https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear)

#### Dropout Layers
* [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)

#### Sparse Layers
* [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)

#### Utilities

* [nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence)
* [nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence)
