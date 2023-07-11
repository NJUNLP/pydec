# PYDEC.NN.FUNCTIONAL.TANH
> pydec.nn.functional.tanh(input, *, ref=None) →  {{pydec_Composition}}

Applies element-wise, $\text{Tanh}(x)=\tanh (x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}$.

This is a nonlinear one variable function and the invocation is dispatched to the currently enabled decomposition algorithm. See [Decompose Activation Functions](decompose-activation-functions.md) for details.

See [torch.nn.Tanh()](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh) for details about `tanh`.