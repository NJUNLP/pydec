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

PyDec has some built-in strategies to decompose bias, and they mostly calculate $p_i$ based on the value of $c_i$. By default, PyDec just adds bias to residual component without performing any bias decomposition. More details about the bias decomposition can be found here (TODO).

## Using Bias Decomposition

Usually you do not need to call an explicit interface to use bias decomposition. PyDec overloads the [addition operator](TODO), whenever `pydec.Composition` is added with `torch.Tensor`, PyDec recognizes it automatically as bias addition and calls the configured bias decomposition.

## Configuring Bias Decomposition
TODO

## Explicit Interface
TODO

## Customizing Bias Decomposition
TODO