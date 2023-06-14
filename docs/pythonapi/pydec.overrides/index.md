# PYDEC.OVERRIDES

This module exposes the various helper functions of handling [`__torch_function__` protocol](https://pytorch.org/docs/stable/notes/extending.html#extending-torch) in PyDec. See [Compatibility with PyTorch](compatibility-with-pytorch.md) for more details on our Torch API dispatching, and see [Extending PyDec](extending-pydec.md) to intervene in the dispatching.


| API                                                                                              | Description                                                                                                           |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.overrides.is_registered short with_parentheses:false{{/auto_link}}           | Returns *True* if the `torch_function` is overridden by PyDec.                                                        |
| {{#auto_link}}pydec.overrides.register_torch_function short with_parentheses:false{{/auto_link}} | The decorator to register a customized function to handle invocations of `torch_function` on {{{pydec_Composition}}}. |