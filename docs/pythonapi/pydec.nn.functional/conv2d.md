# PYDEC.NN.FUNCTIONAL.CONV2D
> pydec.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) →  {{pydec_Composition}}

Applies a 2D convolution over an input image composed of several input planes.

The convolution operation is applied to each component of the input composition. Since the convolution operator is linear, the success of the [tracing](variable-tracing.md) is guaranteed.

See [torch.nn.functional.conv2d()](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d) for details and output shape.