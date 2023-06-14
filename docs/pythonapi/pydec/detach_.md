# PYDEC.DETACH_
> pydec.detach_(input) →  {{{pydec_Composition}}}

Detaches the composition from the graph that created it, making it a leaf. Views cannot be detached in-place.

This method also affects forward mode AD gradients and the result will never have forward mode AD gradients.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.