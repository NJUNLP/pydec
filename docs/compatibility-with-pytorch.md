# Compatibility with PyTorch

## Torch API dispatch mechanism

To minimize the overhead of applying PyDec to pre-defined models. We employ [`__torch_function__` mechanism](https://pytorch.org/docs/stable/notes/extending.html#extending-torch) to dispatch the torch API invocations to the corresponding pydec API implementations. For example, `torch.mul(composition, 2)` is equivalent to `pydec.mul(composition, 2)`.

Thus, for a given model:
```python
import torch
import pydec

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(4, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

tinymodel = TinyModel()
```

It is safe to input a tensor or a composition, the latter producing a decomposition of the model output:
```python
x = torch.rand(4)
out = tinymodel(x)

c = pydec.zeros(x.size(), c_num=x.size(0))
c = pydec.diagonal_init(c, src=x, dim=0)
c_out = tinymodel(c)
```

To get the list functions that are overridable via `__torch_function__`, see [`torch.overrides.get_overridable_functions()`](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_overridable_functions).

To check if a torch API is overridden by pydec, use {{#auto_link}}pydec.overrides.is_registered{{/auto_link}}.

## Incompatible operations

There are some cases PyDec cannot handle automatically, for example, it is illegal to calculate the product of two compositions. In Transformer, these problems occur in Layer Normalization and Attention layer. We propose the method to linearize them in our paper. For Layer Normalization, the linearization has been integrated into the [`torch.nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm) module and the [`torch.nn.functional.layer_norm`](https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html#torch.nn.functional.layer_norm) function, but you still need to linearize the Attention layer and third-party implementations of Layer Normalization manually.

Here is an example of a manually linearized LayerNorm module:
```python
def c2t(input: Union[pydec.Composition, torch.Tensor]):
    if isinstance(input, pydec.Composition):
        return input.c_sum()
    else:
        return input

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        t_hidden_states = c2t(hidden_states)
        variance = t_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
```

See our [examples](https://github.com/DoubleVII/pydec/tree/master/examples) for implementations in real-world models.