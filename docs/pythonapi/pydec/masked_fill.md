# PYDEC.MASKED_FILL
> pydec.masked_fill(input, mask, value) →  {{{pydec_Composition}}}

Fills elements of each component in `input` composition with `value` where `mask` is *True*. The shape of mask `must` be [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with the shape of the input composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **mask** (*BoolTensor*) – the boolean mask.
* **value** (*{{{python_float}}}*) – the value to fill in with.