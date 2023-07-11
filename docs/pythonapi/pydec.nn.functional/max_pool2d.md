# PYDEC.NN.FUNCTIONAL.MAX_POOL2D
> pydec.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) →  {{pydec_Composition}}

Applies a 2D max pooling over an input signal composed of several input planes.

The pooling operation is first applied to the `input.recovery` and gets the indices, and then the elements in each component are selected according to the indices to construct the output composition.

See [torch.nn.functional.max_pool2d()](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d) for details about `max_pool2d`.