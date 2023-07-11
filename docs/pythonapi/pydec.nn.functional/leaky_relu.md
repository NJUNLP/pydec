# PYDEC.NN.FUNCTIONAL.LEAKY_RELU
> pydec.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False, *, ref=None) →  {{pydec_Composition}}

Applies element-wise, $\text{LeakyReLU}(x)=\max (0,x) + \text{negative\_slope} * \min (0, x)$.

This is a nonlinear one variable function and the invocation is dispatched to the currently enabled decomposition algorithm. See [Decompose Activation Functions](decompose-activation-functions.md) for details.

See [torch.nn.LeakyReLU()](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU) for details about `leaky_relu`.