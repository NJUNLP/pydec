---
title: "pydec.bias_decomposition"
description: API for the module pydec.bias_decomposition
---

# PYDEC.BIAS_DECOMPOSITION

{% include codelink.html name="pydec.bias_decomposition" path="pythonapi/pydec.bias_decomposition" %} is a package implementing various bias decomposition algorithms. In addition, we provide interfaces to register user-defined algorithms.

## How to use bias decomposition algorithm

If you have specified a bias decomposition algorithm for PyDec, PyDec will automatically run the bias decomposition when calling addition operator or {% include codelink.html name="pydec.add()" path="pythonapi/pydec/add" %}.


### Configuring bias decomposition

By default, PyDec does not perform any bias decomposition, but adds the bias directly to `Composition._residual`. In this case, Pydec uses the algorithm named {% include codelink.html name="none" path="pythonapi/pydec.bias_decomposition/none" %}.

To specify the default bias decomposition algorithm, use {% include codelink.html name="pydec.set_bias_decomposition_func" path="pythonapi/pydec/set_bias_decomposition_func" %}.

To use the specified bias decomposition algorithm in a local context, use {% include codelink.html name="pydec.using_bias_decomposition_func" path="pythonapi/pydec/using_bias_decomposition_func" %}.

To disable bias decomposition locally, use {% include codelink.html name="pydec.no_bias_decomposition" path="pythonapi/pydec/no_bias_decomposition" %}

### Specify the arguments for bias decomposition

Some bias decomposition functions provide configurable hyperparameters, but the arguments cannot be passed explicitly when using the addition operator.

{% include codelink.html name="pydec.add()" path="pythonapi/pydec/add" %} supports passing in custom keyword arguments.

Example:
```
>>> t = torch.tensor([[1,1],[1,2]]).float() 
>>> c = pydec.Composition(t)
>>> c
composition 0:
tensor([1., 1.])
composition 1:
tensor([1., 2.])
residual:
tensor([0., 0.])
>>> pydec.set_bias_decomposition_func("norm_decomposition")
>>> c.add(1, p=2)
"""
composition 0:
tensor([1.3874, 1.3874])
composition 1:
tensor([1.6126, 2.6126])
residual:
tensor([0., 0.])
"""
>>> c.add(1, p=float("inf"))
"""
composition 0:
tensor([1.3333, 1.3333])
composition 1:
tensor([1.6667, 2.6667])
residual:
tensor([0., 0.])
"""
```

To set the default arguments of the bias decomposition algorithm, use {% include codelink.html name="pydec.set_bias_decomposition_args" path="pythonapi/pydec/set_bias_decomposition_args" %}.

To use the specified arguments of the bias decomposition algorithm in a local context, use {% include codelink.html name="pydec.using_bias_decomposition_args" path="pythonapi/pydec/using_bias_decomposition_args" %}.