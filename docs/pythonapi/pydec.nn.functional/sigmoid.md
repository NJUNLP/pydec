# PYDEC.NN.FUNCTIONAL.SIGMOID
> pydec.nn.functional.sigmoid(input, *, ref=None) →  {{pydec_Composition}}

Applies the element-wise function $\text{Sigmoid}(x)=\frac{1}{1+\exp(-x)}$.

This is a nonlinear one variable function and the invocation is dispatched to the currently enabled decomposition algorithm. See [Decompose Activation Functions](decompose-activation-functions.md) for details.

See [torch.nn.Sigmoid()](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid) for details about `sigmoid`.