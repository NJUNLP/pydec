---
title: "Auto Tracing"
description: Auto Tracing
---
# Auto Tracing

To obtain the decomposition, you need to manually keep a trace of the tensor operations in the forward function of the model, i.e., apply the same operations to the corresponding Composition. This requires you to add additional code to the forward function to complete the forward process for the Composition, although the code is almost identical to that of the tensor.

The autotracing module automatically keeps trace of the tensor operations in the forward, and for simple models you can perform the decomposition without modifying any code; for complex models, you may need to add a small amount of code to perform special processing of the inputs during the decomposition. In any case, the original functionality of the model (performing tensor calculations) is preserved, and you can easily switch between forward calls and decompositions.

To do automatic tracing, first call {% include codelink.html name="pydec.autotracing.compile()" path="pythonapi/pydec.autotracing/compile" %} to compile your model, which returns a new model that wraps your model.

Suppose you have a model named `model`, which should be a `torch.nn.Module`, to compile your model:
```python
>>> model
"""
my_model(
  (linear): Sequential(
    (0): Linear(in_features=3, out_features=3, bias=True)
    (1): Linear(in_features=3, out_features=2, bias=True)
  )
  (param): ParameterList(  (0): Parameter containing: [torch.FloatTensor of size 2])
  (submodel): sub_module(
    (linear): Linear(in_features=4, out_features=3, bias=True)
    (identity): Identity()
  )
)
"""
>>> model = pydec.autotracing.compile(model)
```

Initialize a Composition input with a tensor input:
```python
>>> input = torch.rand((4,))
"""
tensor([0.6853, 0.1793, 0.2301, 0.4074])
"""
>>> c_input = pydec.Composition(input.size(), len(input)).to(input)
>>> c_input = pydec.diagonal_init(c_input, input, dim=0)
>>> c_input
"""
composition 0:
tensor([0.6853, 0.0000, 0.0000, 0.0000])
composition 1:
tensor([0.0000, 0.1793, 0.0000, 0.0000])
composition 2:
tensor([0.0000, 0.0000, 0.2301, 0.0000])
composition 3:
tensor([0.0000, 0.0000, 0.0000, 0.4074])
residual:
tensor([0., 0., 0., 0.])
"""
```

To perform the decomposition:
```python
>>> model(c_input)
"""
composition 0:
tensor([ 0.0310, -0.0436], grad_fn=<UnbindBackward0>)
composition 1:
tensor([ 0.0009, -0.0013], grad_fn=<UnbindBackward0>)
composition 2:
tensor([ 0.0236, -0.0334], grad_fn=<UnbindBackward0>)
composition 3:
tensor([ 0.0266, -0.0376], grad_fn=<UnbindBackward0>)
residual:
tensor([0.2902, 0.5666], grad_fn=<AddBackward0>)
"""
>>> model(c_input).c_sum()
"""
tensor([0.3722, 0.4507], grad_fn=<AddBackward0>)
"""
```

To perform the original forward calculation, use `model.trace()` to switch modes:
```python
>>> model.trace(False)
>>> model(input)
"""
tensor([0.3722, 0.4507], grad_fn=<AddBackward0>)
"""
```