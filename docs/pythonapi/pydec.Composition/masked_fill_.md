# PYDEC.COMPOSITION.MASKED_FILL_
> Composition.masked_fill_(mask, value) →  {{{pydec_Composition}}}

Fills elements of each component (including residual) of `self` composition with value where `mask` is True. The shape of `mask` must be [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with the shape of the underlying composition.

**Parameters:**

* **mask** (*BoolTensor*) – the boolean mask.
* **value** (*{{{torch_Tensor}}} or Number*) –  the value to fill with.