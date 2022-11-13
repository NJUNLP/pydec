---
title: "pydec.bias_decomposition"
description: API for the module pydec.bias_decomposition
---

# PYDEC.BIAS_DECOMPOSITION

{% include codelink.html name="pydec.bias_decomposition" path="pythonapi/pydec.bias_decomposition" %} is a package implementing various bias decomposition algorithms. In addition, we provide interfaces to register user-defined algorithms.

## How to use bias decomposition algorithm

If you have specified a bias decomposition algorithm for PyDec, PyDec will automatically run the bias decomposition when calling addition operator or {% include codelink.html name="pydec.add()" path="pythonapi/pydec/add" %}.


### Configuring bias decomposition

By default, PyDec does not perform any bias decomposition, but adds the bias directly to `Composition._residual`. In this case, Pydec uses the algorithm named {% include codelink.html name="none" path="pythonapi/pydec.bias_decomposition/_none_decomposition" %}.

To specify the default bias decomposition algorithm, use {% include codelink.html name="pydec.set_bias_decomposition_func()" path="pythonapi/pydec/set_bias_decomposition_func" %}.

To use the specified bias decomposition algorithm in a local context, use {% include codelink.html name="pydec.using_bias_decomposition_func()" path="pythonapi/pydec/using_bias_decomposition_func" %}.

To disable bias decomposition locally, use {% include codelink.html name="pydec.no_bias_decomposition()" path="pythonapi/pydec/no_bias_decomposition" %}

To get the currently enabled bias decomposition function, use {% include codelink.html name="pydec.get_bias_decomposition_func()" path="pythonapi/pydec/get_bias_decomposition_func" %}.
Similarly, use {% include codelink.html name="pydec.get_bias_decomposition_name()" path="pythonapi/pydec/get_bias_decomposition_name" %} to get the registered name of this function.

### Specify the arguments for bias decomposition

Some bias decomposition functions provide configurable hyperparameters, but the arguments cannot be passed explicitly when using the addition operator.

{% include codelink.html name="pydec.add()" path="pythonapi/pydec/add" %} supports passing in customized keyword arguments.

Example:
```python
>>> t = torch.tensor([[1,1],[1,2]]).float() 
>>> c = pydec.Composition(t)
>>> c
composition 0:
tensor([1., 1.])
composition 1:
tensor([1., 2.])
residual:
tensor([0., 0.])
>>> pydec.set_bias_decomposition_func('norm_decomposition')
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

A more general approach is to specify the default arguments for bias decomposition. When bias decomposition is called via the addition operator, PyDec passes the configured arguments to the function defining the algorithm via keyword arguments.

To set the default arguments of the bias decomposition algorithm, use {% include codelink.html name="pydec.set_bias_decomposition_args()" path="pythonapi/pydec/set_bias_decomposition_args" %}.

To use the specified arguments of the bias decomposition algorithm in a local context, use {% include codelink.html name="pydec.using_bias_decomposition_args()" path="pythonapi/pydec/using_bias_decomposition_args" %}.

To get the currently set bias decomposition arguments, use {% include codelink.html name="pydec.get_bias_decomposition_args()" path="pythonapi/pydec/get_bias_decomposition_args" %}.

## Customizing bias decomposition

To use a customized bias decomposition algorithm, you need to register your function using {% include codelink.html name="pydec.bias_decomposition.register_bias_decomposition_func" path="pythonapi/pydec.bias_decomposition#register_bias_decomposition_func" %} decorator.

{% include function.html content="pydec.bias_decomposition.register_bias_decomposition_func(name)" %}

Register a bias decomposition function named `name`, cannot be the same name of the already registered algorithms.

Take the decomposition described below as an example:

<blockquote>
Assume that $h^\prime=h+b$ and $h$ is denoted as the sum of $m$ components, i.e., $h=c_1+\cdots+c_m$. Then $b$ is decomposed into $m$ parts and added to each of the $m$ components:

$$
\begin{split}
b=&p_1+\cdots+p_m,\newline
c^\prime_i=&c_i+p_i.
\end{split}
$$

The decomposition of $h^\prime$ was thus obtained as $h^\prime=c^\prime_1+\cdots+c^\prime_m$
</blockquote>

Your bias decomposition function should have a similar signature like:
```python
@pydec.bias_decomposition.register_bias_decomposition_func(your_customized_name)
def customized_decomposition(bias: Union[Number, Tensor], context: Composition) -> Composition:
    ...
```

* **bias** - The bias to be decomposed, i.e., $b$.
* **context** - The composition context of decomposition, i.e., $c_1,\cdots,c_m$.
* Other arguments can be defined as keyword arguments after the `bias` and `context` arguments. To specify these parameters, see [Specify the arguments for bias decomposition](#specify-the-arguments-for-bias-decomposition).

Return the composition obtained by bias decomposition, i.e., $p_1,\cdots,p_m$. The returned composition should be the same shape as the `context` composition and have the same number of components. Also, the sum of all components should be equal to `bias`.

{% include function.html content="Parameters:" %}

* **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) - The registered name of the algorithm.


## Algorithms

| API                                                                                                                                        | Registered Name                  | Description                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {% include codelink.html name="_none_decomposition" path="pythonapi/pydec.bias_decomposition/_none_decomposition" %}                       | *none*                           | No decomposition is performed on `bias`.                                                                                                                                                                                                                                                                      |
| {% include codelink.html name="abs_decomposition" path="pythonapi/pydec.bias_decomposition/abs_decomposition" %}                           | *abs_decomposition*              | The decomposition based on the absolute value of each component in the `context`.                                                                                                                                                                                                                             |
| {% include codelink.html name="norm_decomposition" path="pythonapi/pydec.bias_decomposition/norm_decomposition" %}                         | *norm_decomposition*             | The decomposition based on the vector norm of each component in the `context`.                                                                                                                                                                                                                                |
| {% include codelink.html name="sign_decomposition" path="pythonapi/pydec.bias_decomposition/sign_decomposition" %}                         | *sign_decomposition*             | The decomposition based on the value of each component in the `context`.                                                                                                                                                                                                                                      |
| {% include codelink.html name="sign_decomposition_threshold" path="pythonapi/pydec.bias_decomposition/sign_decomposition_threshold" %} | *sign_decomposition_threshold* | Similar to {% include codelink.html name="sign_decomposition" path="pythonapi/pydec.bias_decomposition/sign_decomposition" %}, the region of unstable values is determined by a threshold value.                                                                                                          |
| {% include codelink.html name="hybrid_decomposition" path="pythonapi/pydec.bias_decomposition/hybrid_decomposition" %}                     | *hybrid_decomposition*           | Similar to {% include codelink.html name="hybrid_decomposition" path="pythonapi/pydec.bias_decomposition/hybrid_decomposition" %}, but switch to {% include codelink.html name="abs_decomposition" path="pythonapi/pydec.bias_decomposition/abs_decomposition" %} algorithm in the region of unstable values. |
| {% include codelink.html name="hybrid_decomposition_threshold" path="pythonapi/pydec.bias_decomposition/hybrid_decomposition_threshold" %} | *hybrid_decomposition_threshold* | Similar to {% include codelink.html name="hybrid_decomposition" path="pythonapi/pydec.bias_decomposition/hybrid_decomposition" %}, the region of unstable values is determined by a threshold value.                                                                                                          |
