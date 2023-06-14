# PYDEC.ISINF
> pydec.isinf(input) →  {{{pydec_Composition}}}

Tests if each element of every component in `input` is infinite (positive or negative infinity) or not.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

Example:
```python
>>> c = pydec.Composition(torch.tensor([[1, float('inf')], [float('-inf'), float('nan')]]))
>>> pydec.isinf(c)
"""
composition{
  components:
    tensor([False,  True]),
    tensor([ True, False]),
  residual:
    tensor([False, False])}
"""
```