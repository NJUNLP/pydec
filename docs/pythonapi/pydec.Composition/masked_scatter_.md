# PYDEC.COMPOSITION.MASKED_SCATTER_
> Composition.masked_scatter_(mask, source) →  {{{pydec_Composition}}}

Copies elements from `source` into each component (including residual) of `self` composition at positions where the `mask` is True. Elements from `source` are copied into `self` starting at position 0 of `source` and continuing in order one-by-one for each occurrence of `mask` being True. The shape of `mask` must be [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with the shape of the underlying composition. The components in `source` should have at least as many elements as the number of ones in `mask`.

**Parameters:**

* **mask** (*BoolTensor*) – the boolean mask.
* **source** (*{{{torch_Tensor}}}*) – the tensor to copy from.

?> The `mask` operates on the `self` composition, not on the given source tensor.

Example:
```python
>>> c = pydec.Composition(torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
>>> c
"""
composition{
  components:
    tensor([0, 0, 0, 0, 0]),
    tensor([0, 0, 0, 0, 0]),
  residual:
    tensor([0, 0, 0, 0, 0])}
"""
>>> mask = torch.tensor([0, 0, 0, 1, 1])
>>> source = torch.tensor([0, 1, 2, 3, 4])
>>> c.masked_scatter_(mask, source)
"""
composition{
  components:
    tensor([0, 0, 0, 0, 1]),
    tensor([0, 0, 0, 2, 3]),
  residual:
    tensor([0, 0, 0, 0, 1])}
"""