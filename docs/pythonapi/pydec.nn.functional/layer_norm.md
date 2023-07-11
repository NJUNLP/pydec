# PYDEC.NN.FUNCTIONAL.LAYER_NORM
> pydec.nn.functional.layer_norm(input, *, ref=None) →  {{pydec_Composition}}

Applies Layer Normalization for last certain number of dimensions.

Since Layer Normalization is a nonlinear function, we apply the linearization trick to linearize it, i.e., the operation of dividing by the standard deviation is considered as linear scaling. In our implementation, the standard deviation is converted to a tensor without being traced by PyDec.

$$
y_i=\frac{x_i-E[x_i]}{\sqrt{\text{Var}[x]+\epsilon}} * \gamma + \beta,
$$
where $x_i$ is the $i$-th component of $x$.

See [torch.nn.LayerNorm()](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm) for details about `layer_norm`.