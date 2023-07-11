# PYDEC.LT
> pydec.lt(input, other, *, out=None) →  {{{pydec_Composition}}}

Computes `input.recovery < other.recovery` element-wise.

The second argument can be a number, a tensor, or a composition whose shape is [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with the first argument. If it is a composition, use its recovery for comparison.


**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the composition to compare.
* **other** (*{{{pydec_Composition}}}, {{{torch_Tensor}}}, or {{{python_float}}}*) – the composition, tensor, or value to compare.

**Keyword Arguments:**
* **out** (*{{{torch_Tensor}}}, optional*) - the output tensor.


Example:
```python
>>> c1 = pydec.Composition(torch.tensor([[1, 2], [3, 4]]))
>>> c2 = pydec.Composition(torch.tensor([[2, 2], [2, 5]]))
>>> pydec.lt(c1, c2)
"""
tensor([False,  True])
"""
```