# PYDEC.EMPTY_INDICES
> pydec.empty_indices(*size, c_num, *, dtype=torch.long, device=None) →  {{{pydec_IndexComposition}}}

Returns a uninitialized index composition, with the shape and the component number defined by the variable argument `size` and `c_num`, respectively.

?> This is the method specifically used to create {{#auto_link}}pydec.IndexComposition short with_parentheses:false{{/auto_link}}, which can safely input sparse layers ([Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) and [EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag)) and return the composition of the indexed tensors.

**Parameters:**

* **size** (*{{{python_int}}}...*) - a sequence of integers defining the shape of the output composition. Can be a variable number of arguments or a collection like a list or tuple. You must use the keyword argument to specify `c_num` if `size` is specified by a variable number of arguments.
* **c_num** (*{{{python_int}}}*) - the number of components of the output composition.

**Keyword Arguments:**

* **dtype** (*{{{torch_dtype}}}, optional*) – the desired data type of returned composition, must be either `torch.int` or `torch.long`. Default: `torch.long`.
* **device** (*{{{torch_device}}}, optional*) – the desired device of returned composition. Default: if **None**, uses the current device for the default tensor type (see {{{torch_set_default_tensor_type}}}). **device** will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.