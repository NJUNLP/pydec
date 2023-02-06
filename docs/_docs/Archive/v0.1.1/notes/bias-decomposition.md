---
title: "Bias Decomposition"
tags: 
 - "bias decomposition"
description: Bias Decomposition
---
# Bias Decomposition

In order to reallocate bias term, PyDec will assign them to components other than residual whenever it encounters bias addition operation.

Assume that $h^\prime=h+b$ and $h$ is denoted as the sum of $m$ components, i.e., $h=c_1+\cdots+c_m$. Then $b$ is decomposed into $m$ parts and added to each of the $m$ components:

$$
\begin{split}
b=&p_1+\cdots+p_m,\newline
c^\prime_i=&c_i+p_i.
\end{split}
$$

The decomposition of $h^\prime$ was thus obtained as $h^\prime=c^\prime_1+\cdots+c^\prime_m$

PyDec has some built-in strategies to decompose bias, and they mostly calculate $p_i$ based on the value of $c_i$. By default, PyDec just adds bias to residual component without performing any bias decomposition.

## Using Bias Decomposition

Usually you do not need to call an explicit interface to use bias decomposition. PyDec overloads the addition operator (TODO), whenever `pydec.Composition` is added with `torch.Tensor`, PyDec recognizes it automatically as bias addition and calls the configured bias decomposition.

## Configuring Bias Decomposition

By default, PyDec does not perform any bias decomposition, but adds the bias directly to `Composition._residual`. To use the specified bias decomposition, call `pydec.set_bias_decomposition_func` and pass in the name of the bias decomposition. This will set the global bias decomposition method for PyDec. Here (TODO) is a list of all the built-in strategies and their names.

### Configuring by context

If you need to locally use the specified bias decomposition method in a context, use `pydec.using_bias_decomposition_func` and pass in the name of the bias decomposition to create the context. `pydec.no_bias_decomposition` provides a context in which bias decomposition is not performed.

Example:
```python
pydec.set_bias_decomposition_func('norm_decomposition')
# The code here uses the method called 'norm_decomposition'.
...
with pydec.using_bias_decomposition_func('abs_decomposition'):
    # The code here uses the method called 'abs_decomposition'.
    ...
    with pydec.no_bias_decomposition():
        # The code here does not perform any bias decomposition'.
        ...
```
### Specify the arguments for bias decomposition

Some bias decomposition functions provide configurable hyperparameters, but the arguments cannot be passed explicitly when using the addition operator.

You can avoid calling the addition operator by calling `pydec.add`, which supports passing in custom keyword arguments.

We recommend using `pydec.set_bias_decomposition_args` to set the arguments of the bias decomposition function. We also provide the context manager to locally set parameters, using `pydec.using_bias_decomposition_args` to create contexts.

## Explicit Interface

You can get the bias decomposition function in the current context via `pydec.get_bias_decomposition_func`. In addition, you can get the name of the current bias decomposition function via `pydec.get_bias_decomposition_name`.

## Customizing Bias Decomposition

See {% include doc.html name="Customizing bias decomposition" path="pythonapi/pydec.bias_decomposition/#customizing-bias-decomposition" %}.