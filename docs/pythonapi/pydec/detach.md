# PYDEC.DETACH
> pydec.detach(input) →  {{{pydec_Composition}}}

Returns a new composition, detached from the current graph.

The result will never require gradient.

This method also affects forward mode AD gradients and the result will never have forward mode AD gradients.

?> Returned Tensor shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.