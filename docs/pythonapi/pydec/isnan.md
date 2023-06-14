# PYDEC.ISNAN
> pydec.isnan(input) →  {{{pydec_Composition}}}

Tests if each element of every component in `input` is NaN or not.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.

Example:
```python
>>> c = pydec.Composition(torch.tensor([[1, float('inf')], [float('-inf'), float('nan')]]))
>>> pydec.isnan(c)
"""
composition{
  components:
    tensor([False, False]),
    tensor([False,  True]),
  residual:
    tensor([False, False])}
"""
```