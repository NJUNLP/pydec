# PYDEC.CLONE
> pydec.clone(input, *, memory_format=torch.preserve_format) →  {{{pydec_Composition}}}

Returns a copy of `input`.

?> This function is differentiable, so gradients will flow back from the result of this operation to `input`. To create a composition without an autograd relationship to `input` see {{#auto_link}}pydec.detach{{/auto_link}} .

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.


**Keyword Arguments:**

* **memory_format** (*{{{torch_memory_format}}}, optional*) – the desired memory format of returned composition. Default: **torch.preserve_format**.